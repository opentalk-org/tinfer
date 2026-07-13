import unittest

from tinfer.server.websocket.capability_mapper import map_styletts2_settings
from tinfer.server.websocket.schemas import VoiceSettings


class CapabilityMappingTest(unittest.TestCase):
    def test_maps_supported_voice_controls_and_seed(self) -> None:
        settings = VoiceSettings(
            speed=1.1,
            alpha=None,
            beta=0.6,
            stability=0.8,
            similarity_boost=0.75,
            style=0.4,
            use_speaker_boost=True,
        )
        self.assertEqual(
            map_styletts2_settings(settings, 7),
            {
                "speed": 1.1,
                "style_interpolation_factor": 0.8,
                "alpha": 0.25,
                "beta": 0.6,
                "embedding_scale": 1.4,
                "seed": 7,
            },
        )


if __name__ == "__main__":
    unittest.main()
