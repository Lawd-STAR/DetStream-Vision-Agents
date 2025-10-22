# OpenRouter Plugin

OpenRouter plugin for vision agents. This plugin provides LLM capabilities using OpenRouter's API, which is compatible with the OpenAI API format.

## Installation

```bash
uv pip install vision-agents-plugins-openrouter
```

## Usage

```python
from vision_agents.plugins import openrouter

llm = openrouter.LLM()
```

## Configuration

This plugin uses the OpenAI-compatible API provided by OpenRouter. You'll need to set your OpenRouter API key as an environment variable or pass it directly to the LLM.

