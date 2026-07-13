import unittest

from tinfer.models.impl.styletts2.model.text_config import TextConfig


class TextConfigTests(unittest.TestCase):
    def test_round_trips_ordered_symbols_and_languages(self):
        config = TextConfig.from_dict(
            {
                "symbols": ["$", "a", "ˈ"],
                "supported_languages": ["en-us", "pl"],
                "default_language": "en-us",
            },
            expected_symbol_count=3,
        )

        self.assertEqual(config.symbols, ("$", "a", "ˈ"))
        self.assertEqual(config.supported_languages, ("en-us", "pl"))
        self.assertEqual(config.to_dict()["symbols"], ["$", "a", "ˈ"])

    def test_rejects_default_language_outside_supported_languages(self):
        with self.assertRaisesRegex(ValueError, "default language"):
            TextConfig.from_dict(
                {
                    "symbols": ["$"],
                    "supported_languages": ["pl"],
                    "default_language": "en-us",
                },
                expected_symbol_count=1,
            )


if __name__ == "__main__":
    unittest.main()
