import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class DocumentationLinkTests(unittest.TestCase):
    def test_all_local_markdown_links_resolve(self):
        broken = []
        pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
        for path in ROOT.rglob("*.md"):
            text = path.read_text(encoding="utf-8", errors="replace")
            for match in pattern.finditer(text):
                target = match.group(1).strip()
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                target = target.split("#", 1)[0]
                if not target:
                    continue
                resolved = (path.parent / target).resolve()
                if not resolved.exists():
                    broken.append((str(path.relative_to(ROOT)), target))
        self.assertEqual(broken, [], msg=f"Broken local Markdown links: {broken}")


if __name__ == "__main__":
    unittest.main()
