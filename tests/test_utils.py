import os
import tempfile
import numpy as np
import av
from vision_agents.core.utils.utils import parse_instructions, Instructions
from getstream.video.rtc.track_util import AudioFormat
from vision_agents.core.utils.video_utils import ensure_even_dimensions
from vision_agents.core.edge.types import PcmData


class TestParseInstructions:
    """Test suite for the parse_instructions function."""

    def test_parse_instructions_no_mentions(self):
        """Test parsing text with no @ mentions."""
        text = "This is a simple instruction without any mentions."
        result = parse_instructions(text)

        assert isinstance(result, Instructions)
        assert result.input_text == text
        assert result.markdown_contents == {}

    def test_parse_instructions_single_mention(self):
        """Test parsing text with a single @ mention."""
        text = "Please read @nonexistent.md for more information."
        result = parse_instructions(text)

        assert result.input_text == text
        assert result.markdown_contents == {"nonexistent.md": ""}  # File doesn't exist

    def test_parse_instructions_multiple_mentions(self):
        """Test parsing text with multiple @ mentions."""
        text = "Check @file1.md and @file2.md for details. Also see @guide.md."
        result = parse_instructions(text)

        assert result.input_text == text
        assert result.markdown_contents == {
            "file1.md": "",
            "file2.md": "",
            "guide.md": "",
        }

    def test_parse_instructions_duplicate_mentions(self):
        """Test parsing text with duplicate @ mentions."""
        text = "Read @nonexistent.md and then @nonexistent.md again."
        result = parse_instructions(text)

        assert result.input_text == text
        # Should only include unique filenames
        assert result.markdown_contents == {"nonexistent.md": ""}

    def test_parse_instructions_non_markdown_mentions(self):
        """Test parsing text with @ mentions that are not markdown files."""
        text = "Check @user123 and @file.txt for information."
        result = parse_instructions(text)

        assert result.input_text == text
        # Should only capture .md files
        assert result.markdown_contents == {}

    def test_parse_instructions_mixed_mentions(self):
        """Test parsing text with both markdown and non-markdown @ mentions."""
        text = "Check @user123, @nonexistent.md, and @config.txt for details."
        result = parse_instructions(text)

        assert result.input_text == text
        # Should only capture .md files
        assert result.markdown_contents == {"nonexistent.md": ""}

    def test_parse_instructions_complex_filenames(self):
        """Test parsing text with complex markdown filenames."""
        text = "See @my-file.md, @file_with_underscores.md, and @file-with-dashes.md."
        result = parse_instructions(text)

        assert result.input_text == text
        assert result.markdown_contents == {
            "my-file.md": "",
            "file_with_underscores.md": "",
            "file-with-dashes.md": "",
        }

    def test_parse_instructions_edge_cases(self):
        """Test parsing text with edge cases."""
        # Empty string
        result = parse_instructions("")
        assert result.input_text == ""
        assert result.markdown_contents == {}

        # Only @ symbol
        result = parse_instructions("@")
        assert result.input_text == "@"
        assert result.markdown_contents == {}

        # @ without filename
        result = parse_instructions("Check @ for details")
        assert result.input_text == "Check @ for details"
        assert result.markdown_contents == {}

        # @ with spaces in filename (should not match)
        result = parse_instructions("Check @my file.md for details")
        assert result.input_text == "Check @my file.md for details"
        assert result.markdown_contents == {}

    def test_parse_instructions_case_sensitivity(self):
        """Test that @ mentions with different cases are extracted separately."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files - use different names (not just case variations)
            # because macOS and Windows use case-insensitive filesystems by default
            file1_path = os.path.join(temp_dir, "Guide.md")
            file2_path = os.path.join(temp_dir, "Help.md")

            with open(file1_path, "w", encoding="utf-8") as f:
                f.write("# Guide Content")

            with open(file2_path, "w", encoding="utf-8") as f:
                f.write("# Help Content")

            # Test that the parser correctly extracts both case variations from text
            # even if they refer to the same file on case-insensitive filesystems
            text = "Check @Guide.md and @guide.md and @Help.md for information."
            result = parse_instructions(text, base_dir=temp_dir)

            assert result.input_text == text
            # Parser should extract all mentioned filenames
            assert "Guide.md" in result.markdown_contents
            assert "guide.md" in result.markdown_contents
            assert "Help.md" in result.markdown_contents
            # On case-insensitive systems, Guide.md and guide.md will have same content
            # but the parser still tracks them separately by their @ mention
            assert len(result.markdown_contents["Guide.md"]) > 0
            assert len(result.markdown_contents["Help.md"]) > 0

    def test_parse_instructions_special_characters(self):
        """Test parsing with special characters in filenames."""
        text = "Check @file-1.md, @file_2.md, and @file.3.md for details."
        result = parse_instructions(text)

        assert result.input_text == text
        assert result.markdown_contents == {
            "file-1.md": "",
            "file_2.md": "",
            "file.3.md": "",
        }

    def test_parse_instructions_multiline_text(self):
        """Test parsing multiline text with @ mentions."""
        text = """Please review the following files:
        - @setup.md for installation instructions
        - @api.md for API documentation
        - @troubleshooting.md for common issues
        """
        result = parse_instructions(text)

        assert result.input_text == text
        assert result.markdown_contents == {
            "setup.md": "",
            "api.md": "",
            "troubleshooting.md": "",
        }


class TestInstructions:
    """Test suite for the Instructions dataclass."""

    def test_instructions_initialization(self):
        """Test Instructions dataclass initialization."""
        input_text = "Test instruction"
        markdown_contents = {"file1.md": "# File 1 content"}

        instructions = Instructions(input_text, markdown_contents)

        assert instructions.input_text == input_text
        assert instructions.markdown_contents == markdown_contents

    def test_instructions_empty_markdown_files(self):
        """Test Instructions with empty markdown files dict."""
        input_text = "Simple instruction"
        markdown_contents = {}

        instructions = Instructions(input_text, markdown_contents)

        assert instructions.input_text == input_text
        assert instructions.markdown_contents == {}

    def test_instructions_equality(self):
        """Test Instructions equality comparison."""
        instructions1 = Instructions("test", {"file.md": "content"})
        instructions2 = Instructions("test", {"file.md": "content"})
        instructions3 = Instructions("different", {"file.md": "content"})

        assert instructions1 == instructions2
        assert instructions1 != instructions3


class TestParseInstructionsFileReading:
    """Test suite for file reading functionality in parse_instructions."""

    def test_parse_instructions_with_existing_files(self):
        """Test parsing with actual markdown files that exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test markdown files
            file1_path = os.path.join(temp_dir, "readme.md")
            file2_path = os.path.join(temp_dir, "guide.md")

            with open(file1_path, "w", encoding="utf-8") as f:
                f.write("# README\n\nThis is a test readme file.")

            with open(file2_path, "w", encoding="utf-8") as f:
                f.write("# Guide\n\nThis is a test guide file.")

            text = "Please read @readme.md and @guide.md for information."
            result = parse_instructions(text, base_dir=temp_dir)

            assert result.input_text == text
            assert (
                result.markdown_contents["readme.md"]
                == "# README\n\nThis is a test readme file."
            )
            assert (
                result.markdown_contents["guide.md"]
                == "# Guide\n\nThis is a test guide file."
            )

    def test_parse_instructions_with_mixed_existing_nonexisting_files(self):
        """Test parsing with mix of existing and non-existing files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create only one test file
            file1_path = os.path.join(temp_dir, "readme.md")
            with open(file1_path, "w", encoding="utf-8") as f:
                f.write("# README\n\nThis file exists.")

            text = "Check @readme.md and @nonexistent.md for details."
            result = parse_instructions(text, base_dir=temp_dir)

            assert result.input_text == text
            assert (
                result.markdown_contents["readme.md"] == "# README\n\nThis file exists."
            )
            assert (
                result.markdown_contents["nonexistent.md"] == ""
            )  # Empty for non-existing file

    def test_parse_instructions_with_custom_base_dir(self):
        """Test parsing with custom base directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file in subdirectory
            subdir = os.path.join(temp_dir, "docs")
            os.makedirs(subdir)
            file_path = os.path.join(subdir, "api.md")

            with open(file_path, "w", encoding="utf-8") as f:
                f.write("# API Documentation\n\nThis is the API docs.")

            text = "See @api.md for API information."
            result = parse_instructions(text, base_dir=subdir)

            assert result.input_text == text
            assert (
                result.markdown_contents["api.md"]
                == "# API Documentation\n\nThis is the API docs."
            )

    def test_parse_instructions_file_read_error_handling(self):
        """Test handling of file read errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file that will cause read errors (permission issues, etc.)
            file_path = os.path.join(temp_dir, "readme.md")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("test content")

            # Make file unreadable (this might not work on all systems)
            try:
                os.chmod(file_path, 0o000)  # No permissions

                text = "Read @readme.md for information."
                result = parse_instructions(text, base_dir=temp_dir)

                assert result.input_text == text
                assert (
                    result.markdown_contents["readme.md"] == ""
                )  # Empty due to read error
            finally:
                # Restore permissions for cleanup
                os.chmod(file_path, 0o644)

    def test_parse_instructions_unicode_content(self):
        """Test parsing with unicode content in markdown files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "unicode.md")

            # Write unicode content
            unicode_content = (
                "# Unicode Test\n\nHello 世界! 🌍\n\nThis has émojis and àccénts."
            )
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(unicode_content)

            text = "Check @unicode.md for unicode content."
            result = parse_instructions(text, base_dir=temp_dir)

            assert result.input_text == text
            assert result.markdown_contents["unicode.md"] == unicode_content

    def test_parse_instructions_default_base_dir(self):
        """Test that default base directory is current working directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test file
            file_path = os.path.join(temp_dir, "readme.md")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("# Test readme content")

            # Change to temp directory to test default base_dir
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)

                # This test verifies that when no base_dir is provided, it uses os.getcwd()
                text = "Read @readme.md for information."
                result = parse_instructions(text)  # No base_dir provided

                assert result.input_text == text
                # Content will not be empty since readme.md exists in current directory
                assert "readme.md" in result.markdown_contents
                assert len(result.markdown_contents["readme.md"]) > 0
                assert result.markdown_contents["readme.md"] == "# Test readme content"
            finally:
                # Always restore original directory
                os.chdir(original_cwd)


class TestPcmDataMethods:
    """Test suite for PcmData class methods."""

    def test_pcm_data_from_bytes(self):
        """Test PcmData.from_bytes class method."""
        # Create test audio data (1 second of 16kHz audio)
        test_samples = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
        audio_bytes = test_samples.tobytes()

        pcm_data = PcmData.from_bytes(
            audio_bytes, sample_rate=16000, format=AudioFormat.S16
        )

        assert pcm_data.sample_rate == 16000
        assert pcm_data.format == "s16"
        assert np.array_equal(pcm_data.samples, test_samples)
        assert pcm_data.duration == 1.0  # 1 second

    def test_pcm_data_resample_same_rate(self):
        """Test resampling when source and target rates are the same."""
        test_samples = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
        pcm_data = PcmData(
            samples=test_samples, sample_rate=16000, format=AudioFormat.S16
        )

        resampled = pcm_data.resample(target_sample_rate=16000)

        # Should return the same data
        assert resampled.sample_rate == 16000
        assert np.array_equal(resampled.samples, test_samples)
        assert resampled.format == "s16"

    def test_pcm_data_resample_24khz_to_48khz(self):
        """Test resampling from 24kHz to 48kHz (Gemini use case)."""
        # Create test audio data (1 second of 24kHz audio)
        test_samples = np.random.randint(-32768, 32767, 24000, dtype=np.int16)
        pcm_data = PcmData(
            samples=test_samples, sample_rate=24000, format=AudioFormat.S16
        )

        resampled = pcm_data.resample(target_sample_rate=48000)

        assert resampled.sample_rate == 48000
        assert resampled.format == "s16"
        # Should have approximately double the samples (24k -> 48k)
        # Handle both 1D and 2D arrays
        num_samples = (
            resampled.samples.shape[-1]
            if resampled.samples.ndim > 1
            else len(resampled.samples)
        )
        assert abs(num_samples - 48000) < 100  # Allow some tolerance
        # Duration should be approximately the same
        assert abs(resampled.duration - 1.0) < 0.1

    def test_pcm_data_resample_48khz_to_16khz(self):
        """Test resampling from 48kHz to 16kHz."""
        # Create test audio data (1 second of 48kHz audio)
        test_samples = np.random.randint(-32768, 32767, 48000, dtype=np.int16)
        pcm_data = PcmData(
            samples=test_samples, sample_rate=48000, format=AudioFormat.S16
        )

        resampled = pcm_data.resample(target_sample_rate=16000)

        assert resampled.sample_rate == 16000
        assert resampled.format == "s16"
        # Should have approximately 1/3 the samples (48k -> 16k)
        # Handle both 1D and 2D arrays
        num_samples = (
            resampled.samples.shape[-1]
            if resampled.samples.ndim > 1
            else len(resampled.samples)
        )
        assert abs(num_samples - 16000) < 100  # Allow some tolerance
        # Duration should be approximately the same
        assert abs(resampled.duration - 1.0) < 0.1

    def test_pcm_data_resample_preserves_metadata(self):
        """Test that resampling preserves PTS, DTS, and time_base metadata."""
        test_samples = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
        pcm_data = PcmData(
            samples=test_samples,
            sample_rate=16000,
            format=AudioFormat.S16,
            pts=1000,
            dts=950,
            time_base=0.001,
        )

        resampled = pcm_data.resample(target_sample_rate=48000)

        assert resampled.pts == 1000
        assert resampled.dts == 950
        assert resampled.time_base == 0.001
        assert abs(resampled.pts_seconds - 1.0) < 0.0001
        assert abs(resampled.dts_seconds - 0.95) < 0.0001

    def test_pcm_data_resample_handles_1d_array(self):
        """Test that resampling handles 1D arrays correctly (fixes ndim error)."""
        # Create test audio data (1 second of 24kHz audio) - 1D array
        test_samples = np.random.randint(-32768, 32767, 24000, dtype=np.int16)
        pcm_data = PcmData(
            samples=test_samples, sample_rate=24000, format=AudioFormat.S16
        )

        # This should now work without the ndim error
        resampled = pcm_data.resample(target_sample_rate=48000)

        assert resampled.sample_rate == 48000
        assert resampled.format == "s16"
        # Should have approximately double the samples (24k -> 48k)
        assert abs(len(resampled.samples) - 48000) < 100  # Allow some tolerance
        # Duration should be approximately the same
        assert abs(resampled.duration - 1.0) < 0.1
        # Output should be 1D array
        assert resampled.samples.ndim == 1

    def test_pcm_data_resample_handles_2d_array(self):
        """Test that resampling handles 2D arrays correctly."""
        # Create test audio data (1 second of 24kHz audio) - 2D array (channels, samples)
        test_samples = np.random.randint(-32768, 32767, (1, 24000), dtype=np.int16)
        pcm_data = PcmData(
            samples=test_samples, sample_rate=24000, format=AudioFormat.S16
        )

        # This should work with 2D arrays too
        resampled = pcm_data.resample(target_sample_rate=48000)

        assert resampled.sample_rate == 48000
        assert resampled.format == "s16"
        # Should have approximately double the samples (24k -> 48k)
        assert abs(len(resampled.samples) - 48000) < 100  # Allow some tolerance
        # Duration should be approximately the same
        assert abs(resampled.duration - 1.0) < 0.1
        # Output should be 1D array (flattened)
        assert resampled.samples.ndim == 1

    def test_pcm_data_from_bytes_and_resample_chain(self):
        """Test chaining from_bytes and resample methods (Gemini use case)."""
        # Create test audio data (1 second of 24kHz audio)
        test_samples = np.random.randint(-32768, 32767, 24000, dtype=np.int16)
        audio_bytes = test_samples.tobytes()

        # Chain the methods like in realtime2.py
        pcm_data = PcmData.from_bytes(
            audio_bytes, sample_rate=24000, format=AudioFormat.S16
        )
        resampled_pcm = pcm_data.resample(target_sample_rate=48000)

        assert pcm_data.sample_rate == 24000
        assert resampled_pcm.sample_rate == 48000
        assert resampled_pcm.format == "s16"
        # Should have approximately double the samples (24k -> 48k)
        assert abs(len(resampled_pcm.samples) - 48000) < 100  # Allow some tolerance
        # Duration should be approximately the same
        assert abs(resampled_pcm.duration - 1.0) < 0.1

    def test_pcm_data_resample_av_array_shape_fix(self):
        """Test that fixes the AV library array shape error (channels, samples)."""
        # Create test audio data that would cause the "Expected packed array.shape[0] to equal 1" error
        test_samples = np.random.randint(
            -32768, 32767, 1920, dtype=np.int16
        )  # Small chunk like in the error
        pcm_data = PcmData(
            samples=test_samples, sample_rate=24000, format=AudioFormat.S16
        )

        # This should work without the array shape error
        resampled = pcm_data.resample(target_sample_rate=48000)

        assert resampled.sample_rate == 48000
        assert resampled.format == "s16"
        # Should have approximately double the samples (1920 -> ~3840)
        assert abs(len(resampled.samples) - 3840) < 100  # Allow some tolerance
        # Output should be 1D array
        assert resampled.samples.ndim == 1


class TestEnsureEvenDimensions:
    """Test suite for ensure_even_dimensions function."""
    
    def test_even_dimensions_unchanged(self):
        """Test that frames with even dimensions pass through unchanged."""
        # Create a frame with even dimensions (1920x1080)
        frame = av.VideoFrame(width=1920, height=1080, format="yuv420p")
        
        result = ensure_even_dimensions(frame)
        
        assert result.width == 1920
        assert result.height == 1080
    
    def test_both_dimensions_odd_cropped(self):
        """Test that frames with both odd dimensions are cropped."""
        # Create a frame with both odd dimensions (1921x1081)
        frame = av.VideoFrame(width=1921, height=1081, format="yuv420p")
        
        result = ensure_even_dimensions(frame)
        
        assert result.width == 1920  # Cropped from 1921
        assert result.height == 1080  # Cropped from 1081
    
    def test_timing_information_preserved(self):
        """Test that pts and time_base are preserved after cropping."""
        from fractions import Fraction
        
        # Create a frame with timing information
        frame = av.VideoFrame(width=1921, height=1081, format="yuv420p")
        frame.pts = 12345
        frame.time_base = Fraction(1, 30)
        
        result = ensure_even_dimensions(frame)
        
        assert result.pts == 12345
        assert result.time_base == Fraction(1, 30)

