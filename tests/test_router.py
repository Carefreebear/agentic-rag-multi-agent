import tempfile
import unittest
from pathlib import Path

from src.agentic_rag import AgenticRAGSystem, route_question


class RouterTests(unittest.TestCase):
    def test_routes_domain_question_to_pdf(self):
        self.assertEqual(route_question("How does multi-head attention work in the paper?").route, "pdf")

    def test_routes_fresh_question_to_web(self):
        self.assertEqual(route_question("What are the latest Transformer efficiency results?").route, "web")

    def test_routes_general_question_to_generation(self):
        self.assertEqual(route_question("Explain why clear interfaces matter.").route, "generate")

    def test_route_is_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            system = AgenticRAGSystem(Path("missing.pdf"), trace)
            system.route("Explain attention")
            self.assertIn('"event": "router_decision"', trace.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
