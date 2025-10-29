import os
import logging
from typing import Optional, List, TYPE_CHECKING, Any, Dict
import json
import boto3
from botocore.exceptions import ClientError

from vision_agents.core.llm.llm import LLM, LLMResponseEvent
from vision_agents.core.llm.llm_types import ToolSchema, NormalizedToolCallItem


from vision_agents.core.llm.events import LLMResponseChunkEvent, LLMResponseCompletedEvent
from vision_agents.core.processors import Processor
from . import events
from vision_agents.core.edge.types import Participant

if TYPE_CHECKING:
    from vision_agents.core.agents.conversation import Message


class BedrockLLM(LLM):
    """
    AWS Bedrock LLM integration for Vision Agents.

    Converse docs can be found here:
    https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html

    Chat history has to be manually passed, there is no conversation storage.
    
    Examples:
    
        from vision_agents.plugins import aws
        llm = aws.LLM(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            region_name="us-east-1"
        )
    """

    def __init__(
        self,
        model: str,
        region_name: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
    ):
        """
        Initialize the BedrockLLM class.
        
        Args:
            model: The Bedrock model ID (e.g., "anthropic.claude-3-5-sonnet-20241022-v2:0")
            region_name: AWS region name (default: "us-east-1")
            aws_access_key_id: Optional AWS access key ID
            aws_secret_access_key: Optional AWS secret access key
            aws_session_token: Optional AWS session token
        """
        super().__init__()
        self.events.register_events_from_module(events)
        self.model = model
        self._pending_tool_uses_by_index: Dict[int, Dict[str, Any]] = {}
        
        # Initialize boto3 bedrock-runtime client
        session_kwargs = {"region_name": region_name}
        if aws_access_key_id:
            session_kwargs["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key:
            session_kwargs["aws_secret_access_key"] = aws_secret_access_key
        if aws_session_token:
            session_kwargs["aws_session_token"] = aws_session_token

        if os.environ.get("AWS_BEDROCK_API_KEY"):
            session_kwargs["aws_session_token"] = os.environ["AWS_BEDROCK_API_KEY"]
            
        self.client = boto3.client("bedrock-runtime", **session_kwargs)

        self.region_name = region_name
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

    async def simple_response(
        self,
        text: str,
        processors: Optional[List[Processor]] = None,
        participant: Optional[Participant] = None,
    ):
        """
        Simple response is a standardized way to create a response.
        
        Args:
            text: The text to respond to
            processors: list of processors (which contain state) about the video/voice AI
            participant: optionally the participant object
        
        Examples:
        
            await llm.simple_response("say hi to the user")
        """
        self.logger.debug(f"simple_response called with text: {text[:100]}...")
        registered_tools = self.get_available_functions()
        self.logger.debug(f"Available functions: {len(registered_tools)} - {[t.get('name') for t in registered_tools]}")
        return await self.converse_stream(
            messages=[{"role": "user", "content": [{"text": text}]}]
        )

    async def converse(self, *args, **kwargs) -> LLMResponseEvent[Any]:
        """
        Converse gives full access to the Bedrock Converse API.
        This method wraps the Bedrock method and broadcasts events for agent integration.
        """
        if "modelId" not in kwargs:
            kwargs["modelId"] = self.model

        # Add tools if available
        tools = self.get_available_functions()
        if tools:
            converted_tools = self._convert_tools_to_provider_format(tools)
            kwargs["toolConfig"] = {
                "tools": converted_tools
            }
            self.logger.debug(f"Added {len(tools)} tools to converse request")

        # Combine original instructions with markdown file contents
        enhanced_instructions = self._build_enhanced_instructions()
        if enhanced_instructions:
            kwargs["system"] = [{"text": enhanced_instructions}]

        # Ensure the AI remembers the past conversation
        new_messages = kwargs.get("messages", [])
        if hasattr(self, '_conversation') and self._conversation:
            old_messages = [m.original for m in self._conversation.messages]
            kwargs["messages"] = old_messages + new_messages
            # Add messages to conversation
            normalized_messages = self._normalize_message(new_messages)
            for msg in normalized_messages:
                self._conversation.messages.append(msg)

        try:
            # Capture system message for use in follow-up calls
            system_param = kwargs.get("system")
            
            # Log initial request details
            self.logger.debug(f"Making converse request with {len(kwargs.get('messages', []))} messages")
            if kwargs.get("toolConfig"):
                tool_count = len(kwargs["toolConfig"].get("tools", []))
                self.logger.info(f"Request includes {tool_count} tools")
            
            response = self.client.converse(**kwargs)
            
            # Extract text from response
            text = self._extract_text_from_response(response)
            llm_response = LLMResponseEvent(response, text)
            
            self.logger.debug(f"Initial response text length: {len(text)}")
            
            # Handle tool calls if present
            function_calls = self._extract_tool_calls_from_response(response)
            if function_calls:
                self.logger.debug(f"Extracted {len(function_calls)} tool calls from response")
                for i, fc in enumerate(function_calls):
                    self.logger.debug(f"  Tool call {i+1}: {fc.get('name')} with args: {fc.get('arguments_json')}")
                messages = kwargs["messages"][:]
                
                # Add assistant message from response (contains toolUse blocks)
                assistant_msg_from_response = response.get('output', {}).get('message', {})
                if assistant_msg_from_response:
                    messages.append(assistant_msg_from_response)
                    self.logger.debug("Added assistant message from response to conversation history")
                
                MAX_ROUNDS = 3
                rounds = 0
                seen: set[tuple[str, str, str]] = set()
                current_calls = function_calls
                
                while current_calls and rounds < MAX_ROUNDS:
                    # Execute calls concurrently with dedup
                    triples, seen = await self._dedup_and_execute(current_calls, seen=seen, max_concurrency=8, timeout_s=30)  # type: ignore[arg-type]
                    
                    if not triples:
                        self.logger.warning("No tool execution results despite tool calls")
                        break
                    
                    self.logger.debug(f"Executed {len(triples)} tool calls, processing results")
                    
                    # Build tool result message
                    # According to AWS Bedrock Converse API, toolResult content should use {"text": "..."} format
                    tool_result_blocks = []
                    for tc, res, err in triples:
                        if err:
                            self.logger.error(f"Tool {tc['name']} execution error: {err}")
                            tool_response = str(err)
                        else:
                            # Convert result to string format (AWS expects text, not json content type)
                            if isinstance(res, (dict, list)):
                                tool_response = json.dumps(res)
                            elif isinstance(res, str):
                                tool_response = res
                            else:
                                tool_response = str(res)
                        
                        tool_result_blocks.append({
                            "toolUseId": tc["id"],
                            "content": [{"text": tool_response}],
                        })

                    # Add user message with tool results only
                    # Assistant message was already added from response before the loop
                    user_tool_results_msg = {
                        "role": "user",
                        "content": [{"toolResult": tr} for tr in tool_result_blocks]
                    }
                    messages = messages + [user_tool_results_msg]
                    
                    self.logger.debug(f"Sending {len(tool_result_blocks)} tool results back to model")

                    # Build follow-up request parameters
                    follow_up_kwargs = {
                        "modelId": self.model,
                        "messages": messages,
                        "toolConfig": kwargs.get("toolConfig", {}),
                    }
                    # Include system message if it was provided
                    if system_param:
                        follow_up_kwargs["system"] = system_param
                    
                    # Ask again WITH tools
                    try:
                        self.logger.debug(f"Sending follow-up request with tool results (round {rounds + 1})")
                        follow_up_response = self.client.converse(**follow_up_kwargs)
                    except ClientError as e:
                        self.logger.error(f"AWS Bedrock API error in follow-up call: {e}")
                        error_code = e.response.get('Error', {}).get('Code', 'Unknown') if hasattr(e, 'response') else 'Unknown'
                        self.logger.error(f"Error code: {error_code}, Full error: {str(e)}")
                        raise
                    
                    # Extract new tool calls and text from follow-up response
                    current_calls = self._extract_tool_calls_from_response(follow_up_response)
                    follow_up_text = self._extract_text_from_response(follow_up_response)
                    llm_response = LLMResponseEvent(follow_up_response, follow_up_text)
                    
                    self.logger.debug(f"Follow-up response: {len(current_calls)} tool calls, text length: {len(follow_up_text)}")
                    
                    # Add assistant message from follow-up response if there are more tool calls
                    if current_calls:
                        assistant_msg_from_follow_up = follow_up_response.get('output', {}).get('message', {})
                        if assistant_msg_from_follow_up:
                            messages.append(assistant_msg_from_follow_up)
                            self.logger.debug("Added assistant message from follow-up response to conversation history")
                    
                    # If follow-up already has text and no more tool calls, we're done!
                    if follow_up_text and not current_calls:
                        self.logger.debug("Follow-up response contains text with no tool calls, using as final response")
                        text = follow_up_text
                        break
                    
                    rounds += 1
                
                # Final pass without tools ONLY if we still have pending tool calls
                if current_calls:
                    self.logger.debug("Performing final pass without tools to get text response")
                    final_kwargs = {
                        "modelId": self.model,
                        "messages": messages,
                    }
                    # Include system message if it was provided
                    if system_param:
                        final_kwargs["system"] = system_param
                    
                    try:
                        self.logger.info("Making final pass request without tools")
                        final_response = self.client.converse(**final_kwargs)
                    except ClientError as e:
                        self.logger.error(f"AWS Bedrock API error in final pass: {e}")
                        error_code = e.response.get('Error', {}).get('Code', 'Unknown') if hasattr(e, 'response') else 'Unknown'
                        self.logger.error(f"Error code: {error_code}, Full error: {str(e)}")
                        raise
                    
                    final_text = self._extract_text_from_response(final_response)
                    llm_response = LLMResponseEvent(final_response, final_text)
                    text = final_text
                    self.logger.info(f"Final response text length: {len(final_text)}")
                elif rounds > 0:
                    # We had tool calls but follow-up gave us text, use that
                    text = llm_response.text if llm_response.text else text
            
            # Use final response text if available, otherwise original text
            final_text_for_event = llm_response.text if hasattr(llm_response, 'text') and llm_response.text else text
            original_for_event = llm_response.original if hasattr(llm_response, 'original') and llm_response.original else response
            
            self.logger.info(f"Emitting LLMResponseCompletedEvent with text length: {len(final_text_for_event)}")
            if not final_text_for_event:
                self.logger.warning("Final response text is empty - model may not have responded")
            
            self.events.send(LLMResponseCompletedEvent(original=original_for_event, text=final_text_for_event, plugin_name="aws"))

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown') if hasattr(e, 'response') else 'Unknown'
            error_msg = e.response.get('Error', {}).get('Message', str(e)) if hasattr(e, 'response') else str(e)
            self.logger.error(f"AWS Bedrock API error: {error_code} - {error_msg}")
            self.logger.error(f"Full error details: {e}")
            llm_response = LLMResponseEvent(None, error_msg, exception = e)
        except Exception as e:
            self.logger.error(f"Unexpected error in converse: {type(e).__name__}: {str(e)}", exc_info=True)
            llm_response = LLMResponseEvent(None, f"Unexpected error: {str(e)}", exception = e)
            
        return llm_response

    async def converse_stream(self, *args, **kwargs) -> LLMResponseEvent[Any]:
        """
        Streaming version of converse using Bedrock's ConverseStream API.
        """
        self.logger.info("converse_stream called")
        if "modelId" not in kwargs:
            kwargs["modelId"] = self.model

        # Add tools if available
        tools = self.get_available_functions()
        if tools:
            converted_tools = self._convert_tools_to_provider_format(tools)
            kwargs["toolConfig"] = {
                "tools": converted_tools
            }
            self.logger.info(f"Added {len(tools)} tools to converse_stream request: {[t.get('name') for t in tools]}")
        else:
            self.logger.info("No tools available for converse_stream request")

        # Ensure the AI remembers the past conversation
        new_messages = kwargs.get("messages", [])
        if hasattr(self, '_conversation') and self._conversation:
            old_messages = [m.original for m in self._conversation.messages]
            kwargs["messages"] = old_messages + new_messages
            normalized_messages = self._normalize_message(new_messages)
            for msg in normalized_messages:
                self._conversation.messages.append(msg)

        # Combine original instructions with markdown file contents
        enhanced_instructions = self._build_enhanced_instructions()
        if enhanced_instructions:
            kwargs["system"] = [{"text": enhanced_instructions}]

        try:
            # Capture system message for use in follow-up calls
            system_param = kwargs.get("system")
            
            # Log initial request details
            self.logger.info(f"Making converse_stream request with {len(kwargs.get('messages', []))} messages")
            if kwargs.get("toolConfig"):
                tool_count = len(kwargs["toolConfig"].get("tools", []))
                self.logger.info(f"Request includes {tool_count} tools")
            
            try:
                response = self.client.converse_stream(**kwargs)
                self.logger.info("converse_stream API call succeeded, processing stream...")
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown') if hasattr(e, 'response') else 'Unknown'
                error_msg = e.response.get('Error', {}).get('Message', str(e)) if hasattr(e, 'response') else str(e)
                self.logger.error(f"AWS Bedrock API error in converse_stream: {error_code} - {error_msg}")
                self.logger.error(f"Full error details: {e}", exc_info=True)
                raise
            
            stream = response.get('stream')
            if not stream:
                self.logger.error("converse_stream response has no 'stream' field")
                llm_response = LLMResponseEvent(None, "No stream in response")
                return llm_response
            
            text_parts: List[str] = []
            accumulated_calls: List[NormalizedToolCallItem] = []
            last_event = None
            
            # Process stream
            event_count = 0
            for event in stream:
                event_count += 1
                last_event = event
                self._process_stream_event(event, text_parts, accumulated_calls)
            
            self.logger.info(f"Processed {event_count} stream events, accumulated {len(accumulated_calls)} tool calls, {len(''.join(text_parts))} chars of text")
            
            # Handle multi-hop tool calling
            messages = kwargs["messages"][:]
            MAX_ROUNDS = 3
            rounds = 0
            seen: set[tuple[str, str, str]] = set()
            
            if accumulated_calls:
                self.logger.info(f"Extracted {len(accumulated_calls)} tool calls from stream")
                # Build assistant message from accumulated tool calls
                # This matches the format from AWS response['output']['message']
                assistant_content = []
                for tool_call in accumulated_calls:
                    assistant_content.append({
                        "toolUse": {
                            "toolUseId": tool_call["id"],
                            "name": tool_call["name"],
                            "input": tool_call["arguments_json"],
                        }
                    })
                assistant_msg_from_stream = {
                    "role": "assistant",
                    "content": assistant_content
                }
                messages.append(assistant_msg_from_stream)
                self.logger.debug("Added assistant message from stream to conversation history")
            
            while accumulated_calls and rounds < MAX_ROUNDS:
                triples, seen = await self._dedup_and_execute(accumulated_calls, seen=seen, max_concurrency=8, timeout_s=30)  # type: ignore[arg-type]
                
                if not triples:
                    self.logger.warning("No tool execution results despite tool calls")
                    break
                
                self.logger.debug(f"Executed {len(triples)} tool calls, processing results")
                
                # Build tool result messages
                # According to AWS Bedrock Converse API, toolResult content should use {"text": "..."} format
                tool_result_blocks = []
                for tc, res, err in triples:
                    if err:
                        self.logger.error(f"Tool {tc['name']} execution error: {err}")
                        tool_response = str(err)
                    else:
                        # Convert result to string format (AWS expects text, not json content type)
                        if isinstance(res, (dict, list)):
                            tool_response = json.dumps(res)
                        elif isinstance(res, str):
                            tool_response = res
                        else:
                            tool_response = str(res)
                    
                    tool_result_blocks.append({
                        "toolUseId": tc["id"],
                        "content": [{"text": tool_response}],
                    })

                # Add user message with tool results only
                # Assistant message was already added from stream before the loop
                user_tool_results_msg = {
                    "role": "user",
                    "content": [{"toolResult": tr} for tr in tool_result_blocks]
                }
                messages = messages + [user_tool_results_msg]
                
                self.logger.debug(f"Sending {len(tool_result_blocks)} tool results back to model")

                # Build follow-up request parameters
                follow_up_kwargs = {
                    "modelId": self.model,
                    "messages": messages,
                    "toolConfig": kwargs.get("toolConfig", {}),
                }
                # Include system message if it was provided
                if system_param:
                    follow_up_kwargs["system"] = system_param
                
                # Next round with tools
                follow_up_response = self.client.converse_stream(**follow_up_kwargs)
                
                accumulated_calls = []
                follow_up_text_parts: List[str] = []
                follow_up_stream = follow_up_response.get('stream')
                for event in follow_up_stream:
                    last_event = event
                    self._process_stream_event(event, follow_up_text_parts, accumulated_calls)
                
                # Append follow-up text to main text parts
                if follow_up_text_parts:
                    text_parts.extend(follow_up_text_parts)
                
                self.logger.debug(f"Follow-up response: {len(accumulated_calls)} tool calls, text length: {len(''.join(follow_up_text_parts))}")
                
                # Add assistant message from follow-up stream if there are more tool calls
                if accumulated_calls:
                    # Build assistant message from accumulated tool calls from follow-up
                    follow_up_assistant_content = []
                    for tool_call in accumulated_calls:
                        follow_up_assistant_content.append({
                            "toolUse": {
                                "toolUseId": tool_call["id"],
                                "name": tool_call["name"],
                                "input": tool_call["arguments_json"],
                            }
                        })
                    follow_up_assistant_msg = {
                        "role": "assistant",
                        "content": follow_up_assistant_content
                    }
                    messages.append(follow_up_assistant_msg)
                    self.logger.debug("Added assistant message from follow-up stream to conversation history")
                
                # If follow-up already has text and no more tool calls, we're done!
                if follow_up_text_parts and not accumulated_calls:
                    self.logger.info("Follow-up response contains text with no tool calls, using as final response")
                    break
                
                rounds += 1
            
            # Final pass without tools ONLY if we still have pending tool calls
            if accumulated_calls:
                self.logger.info("Performing final pass without tools to get text response")
                final_kwargs = {
                    "modelId": self.model,
                    "messages": messages,
                }
                # Include system message if it was provided
                if system_param:
                    final_kwargs["system"] = system_param
                
                final_response = self.client.converse_stream(**final_kwargs)
                final_stream = final_response.get('stream')
                final_text_parts: List[str] = []
                for event in final_stream:
                    last_event = event
                    self._process_stream_event(event, final_text_parts, accumulated_calls)
                if final_text_parts:
                    text_parts.extend(final_text_parts)
            
            total_text = "".join(text_parts)
            llm_response = LLMResponseEvent(last_event, total_text)
            self.events.send(LLMResponseCompletedEvent(original=last_event, text=total_text, plugin_name="aws"))
            
        except ClientError as e:
            error_msg = f"AWS Bedrock streaming error: {str(e)}"
            llm_response = LLMResponseEvent(None, error_msg)
            
        return llm_response

    def _process_stream_event(
        self, 
        event: Dict[str, Any], 
        text_parts: List[str],
        accumulated_calls: List[NormalizedToolCallItem]
    ):
        """Process a streaming event from AWS."""
        # Forward the native event
        self.events.send(events.AWSStreamEvent(
            plugin_name="aws",
            event_data=event
        ))
        
        # Handle content block delta (text)
        if 'contentBlockDelta' in event:
            delta = event['contentBlockDelta']['delta']
            if 'text' in delta:
                text_parts.append(delta['text'])
                self.events.send(LLMResponseChunkEvent(
                    plugin_name="aws",
                    content_index=event['contentBlockDelta'].get('contentBlockIndex', 0),
                    item_id="",
                    output_index=0,
                    sequence_number=0,
                    delta=delta['text'],
                ))
        
        # Handle tool use
        if 'contentBlockStart' in event:
            start = event['contentBlockStart'].get('start', {})
            if 'toolUse' in start:
                tool_use = start['toolUse']
                idx = event['contentBlockStart'].get('contentBlockIndex', 0)
                self._pending_tool_uses_by_index[idx] = {
                    "id": tool_use.get('toolUseId', ''),
                    "name": tool_use.get('name', ''),
                    "parts": []
                }
        
        if 'contentBlockDelta' in event:
            delta = event['contentBlockDelta']['delta']
            if 'toolUse' in delta:
                idx = event['contentBlockDelta'].get('contentBlockIndex', 0)
                if idx in self._pending_tool_uses_by_index:
                    input_data = delta['toolUse'].get('input', '')
                    self._pending_tool_uses_by_index[idx]['parts'].append(input_data)
        
        if 'contentBlockStop' in event:
            idx = event['contentBlockStop'].get('contentBlockIndex', 0)
            pending = self._pending_tool_uses_by_index.pop(idx, None)
            if pending:
                buf = "".join(pending["parts"]).strip() or "{}"
                try:
                    args = json.loads(buf)
                except Exception:
                    args = {}
                tool_call_item: NormalizedToolCallItem = {
                    "type": "tool_call",
                    "id": pending["id"],
                    "name": pending["name"],
                    "arguments_json": args
                }
                accumulated_calls.append(tool_call_item)

    def _extract_text_from_response(self, response: Dict[str, Any]) -> str:
        """Extract text content from AWS response."""
        output = response.get('output', {})
        message = output.get('message', {})
        content = message.get('content', [])
        
        text_parts = []
        for item in content:
            if 'text' in item:
                text_parts.append(item['text'])
        
        return "".join(text_parts)

    def _extract_tool_calls_from_response(self, response: Dict[str, Any]) -> List[NormalizedToolCallItem]:
        """Extract tool calls from AWS response."""
        tool_calls = []
        
        output = response.get('output', {})
        if not output:
            self.logger.debug("Response has no 'output' field")
            return tool_calls
            
        message = output.get('message', {})
        if not message:
            self.logger.debug("Response output has no 'message' field")
            return tool_calls
            
        content = message.get('content', [])
        if not content:
            self.logger.debug("Response message has no 'content' field")
            return tool_calls
        
        self.logger.debug(f"Checking {len(content)} content items for tool calls")
        for item in content:
            if 'toolUse' in item:
                tool_use = item['toolUse']
                tool_call: NormalizedToolCallItem = {
                    "type": "tool_call",
                    "id": tool_use.get('toolUseId', ''),
                    "name": tool_use.get('name', ''),
                    "arguments_json": tool_use.get('input', {})
                }
                tool_calls.append(tool_call)
                self.logger.debug(f"Found tool call: {tool_call['name']} (id: {tool_call['id']})")
        
        return tool_calls

    def _convert_tools_to_provider_format(self, tools: List[ToolSchema]) -> List[Dict[str, Any]]:
        """
        Convert ToolSchema objects to AWS Bedrock format.
        
        Args:
            tools: List of ToolSchema objects
            
        Returns:
            List of tools in AWS Bedrock format
        """
        aws_tools = []
        for tool in tools:
            name = tool.get("name", "unnamed_tool")
            description = tool.get("description", "") or ""
            params = tool.get("parameters_schema") or {}
            
            # Normalize to a valid JSON Schema object
            if not isinstance(params, dict):
                params = {}
            
            # Ensure it has the required JSON Schema structure
            if "type" not in params:
                # Extract required fields from properties if they exist
                properties = params if params else {}
                required = list(properties.keys()) if properties else []
                
                params = {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False
                }
            else:
                # Already has type, but ensure additionalProperties is set
                if "additionalProperties" not in params:
                    params["additionalProperties"] = False
            
            # AWS Bedrock Converse API expects toolSpec format with inputSchema.json as a DICT
            aws_tool = {
                "toolSpec": {
                    "name": name,
                    "description": description,
                    "inputSchema": {
                        "json": params  # This is a dict, not a JSON string
                    }
                }
            }
            aws_tools.append(aws_tool)
        return aws_tools

    @staticmethod
    def _normalize_message(aws_messages: Any) -> List["Message"]:
        """Normalize AWS messages to internal Message format."""
        from vision_agents.core.agents.conversation import Message

        if isinstance(aws_messages, str):
            aws_messages = [
                {"content": [{"text": aws_messages}], "role": "user"}
            ]

        if not isinstance(aws_messages, (List, tuple)):
            aws_messages = [aws_messages]

        messages: List[Message] = []
        for m in aws_messages:
            if isinstance(m, dict):
                content_items = m.get("content", [])
                # Extract text from content blocks
                text_parts = []
                for item in content_items:
                    if isinstance(item, dict) and 'text' in item:
                        text_parts.append(item['text'])
                    elif isinstance(item, str):
                        text_parts.append(item)
                content = " ".join(text_parts)
                role = m.get("role", "user")
            else:
                content = str(m)
                role = "user"
            message = Message(original=m, content=content, role=role)
            messages.append(message)

        return messages

