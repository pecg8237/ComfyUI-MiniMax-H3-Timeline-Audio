import sys
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from minimax_h3_long_video.timeline import (
    build_prompt_plan,
    plan_segments,
    prompt_plan_prompts,
    slice_prompt,
)


class TimelineTests(unittest.TestCase):
    def test_author_2026_08_25_plan_for_720_frames_uses_strict_124_ceiling(self):
        segments = plan_segments(720, 39, False, 124)
        self.assertEqual([item.raw_frames for item in segments], [124] * 8 + [56])
        self.assertEqual([item.context_frames for item in segments], [0] + [39] * 8)
        self.assertEqual([item.output_frames for item in segments], [124] + [85] * 7 + [1])
        self.assertEqual(
            [item.output_start for item in segments],
            [0, 124, 209, 294, 379, 464, 549, 634, 719],
        )
        self.assertEqual(sum(item.output_frames for item in segments), 720)
        self.assertTrue(all(item.raw_frames <= 124 for item in segments))
        self.assertTrue(all(item.raw_frames % 17 == 5 for item in segments))

    def test_initial_latent_uses_context_inside_the_first_raw_segment(self):
        segments = plan_segments(170, 39, True, 124)
        self.assertEqual(
            [(item.raw_frames, item.context_frames, item.output_frames) for item in segments],
            [(124, 39, 85), (124, 39, 85)],
        )
        self.assertEqual(segments[0].output_start, 0)

    def test_short_final_segment_is_grid_aligned_and_delivers_exact_length(self):
        segments = plan_segments(360, 39, False, 124)
        self.assertEqual(
            [(item.raw_frames, item.context_frames, item.output_frames) for item in segments],
            [(124, 0, 124), (124, 39, 85), (124, 39, 85), (107, 39, 66)],
        )
        self.assertEqual(sum(item.output_frames for item in segments), 360)

    def test_max_raw_frames_is_not_reversed_to_a_whole_second_window(self):
        segments = plan_segments(240, 39, False, 124)
        self.assertEqual((segments[0].raw_frames, segments[0].output_frames), (124, 124))
        self.assertEqual((segments[1].raw_frames, segments[1].output_frames), (124, 85))

    def test_invalid_plans_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "17k\\+5"):
            plan_segments(240, 39, False, 120)
        with self.assertRaisesRegex(ValueError, "greater than context_frames"):
            plan_segments(240, 39, False, 39)
        with self.assertRaisesRegex(ValueError, "context_frames"):
            plan_segments(240, 30, False, 124)

    def test_prompt_end_uses_generated_frames_after_context(self):
        last = plan_segments(720, 39, False, 124)[-1]
        self.assertAlmostEqual(last.prompt_start_seconds, 719 / 24)
        self.assertAlmostEqual(last.prompt_end_seconds, 736 / 24)

    def test_continuation_clock_places_guide_before_new_shots(self):
        prompt = (
            "[Shot 1] Keep running without stopping. "
            "[Shot 2] At 00:06.000, jump over the rail."
        )
        sliced = slice_prompt(prompt, 124 / 24, 209 / 24, 39 / 24)
        self.assertIn("For the first 1.625 seconds", sliced)
        self.assertIn("At 00:01.625, continue forward", sliced)
        self.assertIn("[Shot 2] At 00:02.458, jump over the rail", sliced)
        self.assertNotIn("Keep running without stopping", sliced)

    def test_segment_without_new_shot_uses_guide_only_not_replayed_shot_prose(self):
        prompt = "[Shot 1] Perform a long spin, kick, landing, grab, and swing sequence."
        sliced = slice_prompt(prompt, 124 / 24, 209 / 24, 39 / 24, 1, 9, 30.0)
        self.assertIn("follow the supplied AV guide exactly", sliced)
        self.assertIn("do not repeat the guided action", sliced)
        self.assertNotIn("spin, kick, landing", sliced)

    def test_global_instructions_and_reference_prefix_remain_in_every_segment(self):
        prompt = """subject_definitions:
<Subject 1> is the woman in <Picture 1>.

detailed_description:
One continuous take.
[Shot 1] Walk forward.
[Shot 2] At 00:07.000, turn left.

[Global Instructions]
Preserve the same identity, costume, camera, and lighting.

overall_soundscape: Continuous surf.
non_diegetic_music: One uninterrupted song.
"""
        sliced = slice_prompt(prompt, 124 / 24, 209 / 24, 39 / 24)
        self.assertIn("<Subject 1> is the woman", sliced)
        self.assertIn("Preserve the same identity", sliced)
        self.assertIn("overall_soundscape: Continuous surf", sliced)
        self.assertIn("non_diegetic_music: One uninterrupted song", sliced)
        self.assertNotIn("[Global Instructions]", sliced)

    def test_integrated_prompt_keeps_common_intro(self):
        prompt = """integrated_multimodal_description:
Use clean cel-shaded anime rendering.
[Shot 1] Begin walking.
[Shot 2] At 00:06.000, look back.
"""
        sliced = slice_prompt(prompt, 124 / 24, 209 / 24, 39 / 24)
        self.assertIn("Use clean cel-shaded anime rendering", sliced)
        self.assertIn("look back", sliced)

    def test_untimed_shots_are_distributed_by_segment(self):
        prompt = " ".join("[Shot {}] action {}.".format(i, i) for i in range(1, 10))
        segments = plan_segments(720, 39, False, 124)
        local = [
            slice_prompt(
                prompt,
                item.prompt_start_seconds,
                item.prompt_end_seconds,
                item.context_frames / 24,
                item.index,
                len(segments),
                30.0,
            )
            for item in segments
        ]
        for index, value in enumerate(local, start=1):
            self.assertIn("action {}".format(index), value)
            self.assertEqual(
                sum("action {}".format(i) in value for i in range(1, 10)),
                1,
            )

    def test_prompt_plan_matches_author_segment_records_and_override(self):
        prompt = (
            "[Shot 1] Start. "
            "[Shot 2] At 00:06.000, continue. "
            "[Shot 3] At 00:10.000, finish."
        )
        override = "[Shot 1] Manual continuation."
        plan = build_prompt_plan(
            prompt, 360, 124, "39", False,
            {"segment_prompt_1": override},
        )
        segments = plan_segments(360, 39, False, 124)
        self.assertEqual(len(plan["segments"]), 4)
        self.assertEqual(plan["segments"][1]["prompt"], override)
        self.assertEqual(
            prompt_plan_prompts(plan, segments, 360, 124, 39, False),
            [entry["prompt"] for entry in plan["segments"]],
        )
        with self.assertRaisesRegex(ValueError, "settings do not match"):
            prompt_plan_prompts(plan, segments, 360, 124, 22, False)

    def test_shot_validation_is_retained(self):
        with self.assertRaisesRegex(ValueError, "sequential starting at 1"):
            slice_prompt("[Shot 1] Start. [Shot 3] At 00:04.000, skip.", 0, 10)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            slice_prompt(
                "[Shot 1] Start. [Shot 2] At 00:05.000, later. "
                "[Shot 3] At 00:04.000, backwards.",
                0,
                10,
            )
        with self.assertRaisesRegex(ValueError, "at most one"):
            slice_prompt(
                "[Shot 1] Start.\n[Global Instructions]\nA.\n"
                "[Global Instructions]\nB.",
                0,
                10,
            )

    def test_freeform_fallback_rebases_timestamps_after_guide(self):
        sliced = slice_prompt(
            "At 00:06.000 the light changes.",
            124 / 24,
            209 / 24,
            39 / 24,
        )
        self.assertIn("00:02.458", sliced)
        self.assertIn("follow the supplied AV guide exactly", sliced)


if __name__ == "__main__":
    unittest.main()
