import sys
import unittest
from pathlib import Path

import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from minimax_h3_long_video.audio_timeline import slice_timeline_audio
from minimax_h3_long_video.timeline import Segment


class AudioTimelineTests(unittest.TestCase):
    def test_slice_includes_context_and_h3_padding(self):
        waveform = torch.arange(1, 301, dtype=torch.float32).reshape(1, 1, -1)
        audio = {"waveform": waveform, "sample_rate": 24, "marker": "preserved"}

        first = Segment(0, 124, 0, 0, 120)
        first_slice = slice_timeline_audio(audio, first)
        self.assertEqual(first_slice["waveform"].shape, (1, 1, 124))
        self.assertTrue(torch.equal(first_slice["waveform"], waveform[..., :124]))
        self.assertEqual(first_slice["marker"], "preserved")

        continuation = Segment(1, 175, 39, 120, 120)
        continuation_slice = slice_timeline_audio(audio, continuation)
        self.assertEqual(continuation_slice["waveform"].shape, (1, 1, 175))
        self.assertEqual(continuation_slice["waveform"][0, 0, 0].item(), 82)
        self.assertEqual(continuation_slice["waveform"][0, 0, -1].item(), 256)

    def test_slice_pads_before_and_after_source(self):
        waveform = torch.arange(1, 101, dtype=torch.float32).reshape(1, 1, -1)
        audio = {"waveform": waveform, "sample_rate": 24}

        initial_context = Segment(0, 73, 39, 0, 24)
        initial_slice = slice_timeline_audio(audio, initial_context)["waveform"]
        self.assertEqual(initial_slice.shape[-1], 73)
        self.assertTrue(torch.equal(initial_slice[..., :39], torch.zeros_like(initial_slice[..., :39])))
        self.assertEqual(initial_slice[0, 0, 39].item(), 1)

        beyond_end = Segment(2, 73, 22, 96, 24)
        end_slice = slice_timeline_audio(audio, beyond_end)["waveform"]
        self.assertEqual(end_slice.shape[-1], 73)
        self.assertEqual(end_slice[0, 0, 0].item(), 75)
        self.assertEqual(end_slice[0, 0, 25].item(), 100)
        self.assertTrue(torch.equal(end_slice[..., 26:], torch.zeros_like(end_slice[..., 26:])))

    def test_rejects_invalid_audio_shape_and_rate(self):
        segment = Segment(0, 124, 0, 0, 120)
        with self.assertRaisesRegex(ValueError, "shape"):
            slice_timeline_audio({"waveform": torch.zeros(10), "sample_rate": 32000}, segment)
        with self.assertRaisesRegex(ValueError, "sample_rate"):
            slice_timeline_audio({"waveform": torch.zeros(1, 1, 10), "sample_rate": 0}, segment)


if __name__ == "__main__":
    unittest.main()
