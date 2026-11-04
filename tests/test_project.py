import json
import re
import subprocess
import unittest
from html.parser import HTMLParser
from pathlib import Path

from games_analytics.build import portable_model, predict_portable
from games_analytics.core import CAT_EXT, NUM_INPUT, NUM_LOG, candidates, input_to_model, load_data, prepare, split

ROOT = Path(__file__).resolve().parents[1]


class ElementInventory(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.external = []
        self.file_links = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if "id" in attributes:
            self.ids.add(attributes["id"])
        for key in ("src", "href"):
            value = attributes.get(key, "")
            if value.startswith(("http:", "https:", "//")):
                self.external.append(value)
            elif value and not value.startswith("#"):
                self.file_links.append(value)


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads((ROOT / "artifacts" / "results.json").read_text(encoding="utf-8"))
        cls.train, cls.test = split(prepare(load_data()))
        selected = cls.payload["report"]["selected"]
        candidate, columns = candidates()[selected]
        cls.fitted = candidate.fit(cls.train[columns], cls.train["Global_Sales"])
        cls.spec = portable_model(cls.fitted, cls.train)

    def test_dataset_and_holdout(self):
        self.assertEqual(len(self.train), 13375)
        self.assertEqual(len(self.test), 3344)
        self.assertTrue(set(self.train.index).isdisjoint(self.test.index))
        self.assertEqual(len(self.payload["report"]["test_models"]), 7)
        self.assertFalse(set(["Name", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]) & set(NUM_LOG + CAT_EXT))

    def test_python_and_browser_predictions_match(self):
        records = [item["record"] for item in self.payload["report"]["examples"]]
        records += [
            {field: None for field in NUM_INPUT + CAT_EXT},
            {"Year_of_Release": 2015, "Critic_Score": 78, "Critic_Count": 0, "User_Score": 7.3, "User_Count": 0, "Platform": "UNKNOWN_PLATFORM", "Genre": "Action", "Rating": "E", "Publisher": "UNKNOWN_PUBLISHER", "Developer": None},
            {"Year_of_Release": 2025, "Critic_Score": 100, "Critic_Count": 1000, "User_Score": 10, "User_Count": 100000, "Platform": "PS4", "Genre": "Sports", "Rating": "E", "Publisher": "Nintendo", "Developer": "Nintendo"},
        ]
        completed = subprocess.run(
            ["node", str(ROOT / "tests" / "parity.cjs")],
            input=json.dumps({"model": self.payload["model"], "records": records}),
            text=True, capture_output=True, check=True, cwd=ROOT,
        )
        browser = json.loads(completed.stdout)
        for record, js in zip(records, browser):
            expected = float(self.fitted.predict(input_to_model(record))[0])
            python, effects = predict_portable(self.spec, record)
            self.assertAlmostEqual(expected, python, places=8)
            self.assertAlmostEqual(expected, js["raw"], places=8)
            self.assertAlmostEqual(max(0, expected), js["clipped"], places=8)
            self.assertAlmostEqual(js["raw"], self.payload["model"]["baseline"] + sum(js["contributions"].values()), places=8)
            self.assertAlmostEqual(python, self.spec["baseline"] + sum(effects.values()), places=8)

    def test_offline_artifact_is_complete(self):
        html = (ROOT / "index.html").read_text(encoding="utf-8")
        self.assertIn("const PAYLOAD=", html)
        self.assertIn("function predict(record)", html)
        self.assertNotIn("/*__DATA__*/", html)
        self.assertNotIn("<script src=", html)
        self.assertNotIn("<link rel=", html)
        inventory = ElementInventory()
        inventory.feed(html)
        self.assertEqual(inventory.external, [])
        self.assertEqual(inventory.file_links, [])
        for field in NUM_INPUT + CAT_EXT:
            self.assertIn(field, inventory.ids)
        script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
        referenced_ids = set(re.findall(r'byId\("([A-Za-z0-9_-]+)"\)', script))
        self.assertEqual(referenced_ids - inventory.ids, set())
        embedded = re.search(r"const PAYLOAD=(\{.*?\});</script>", html, re.DOTALL)
        self.assertIsNotNone(embedded)
        self.assertEqual(json.loads(embedded.group(1))["report"]["selected"], self.payload["report"]["selected"])

    def test_search_and_dynamic_story(self):
        self.assertGreater(len(self.payload["report"]["catalog"]), 3000)
        completed = subprocess.run(["node", str(ROOT / "tests" / "ui.cjs")], text=True, capture_output=True, check=True, cwd=ROOT)
        self.assertIn("OK", completed.stdout)


if __name__ == "__main__":
    unittest.main()
