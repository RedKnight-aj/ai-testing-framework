"""
Reporters - Generate evaluation reports
"""

import json
from typing import List, Dict, Any
from pathlib import Path
from abc import ABC, abstractmethod

from .runner import EvaluationResult


class BaseReporter(ABC):
    """Base reporter interface."""
    
    @abstractmethod
    def generate(self, results: List[EvaluationResult]) -> str:
        """Generate report from results."""
        pass
    
    @abstractmethod
    def save(self, results: List[EvaluationResult], path: str):
        """Save report to file."""
        pass


class JSONReporter(BaseReporter):
    """Generate JSON reports."""
    
    def generate(self, results: List[EvaluationResult]) -> str:
        """Generate JSON report."""
        data = {
            "summary": {
                "total_tests": len(results),
                "passed": sum(1 for r in results if r.passed),
                "failed": sum(1 for r in results if not r.passed),
                "average_score": sum(r.score for r in results) / len(results) if results else 0,
            },
            "results": [r.to_dict() for r in results],
        }
        return json.dumps(data, indent=2)
    
    def save(self, results: List[EvaluationResult], path: str):
        """Save JSON report to file."""
        content = self.generate(results)
        Path(path).write_text(content)
        print(f"JSON report saved to: {path}")


class HTMLReporter(BaseReporter):
    """Generate HTML reports."""
    
    def generate(self, results: List[EvaluationResult]) -> str:
        """Generate HTML report."""
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        avg_score = sum(r.score for r in results) / total if total else 0
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>AI Testing Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background: #f5f5f5; padding: 20px; border-radius: 8px; }}
        .passed {{ color: green; }}
        .failed {{ color: red; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .score {{ font-weight: bold; }}
    </style>
</head>
<body>
    <h1>🧪 AI Testing Report</h1>
    <div class="summary">
        <h2>Summary</h2>
        <p>Total Tests: <strong>{total}</strong></p>
        <p class="passed">Passed: <strong>{passed}</strong></p>
        <p class="failed">Failed: <strong>{failed}</strong></p>
        <p>Average Score: <strong>{avg_score:.2%}</strong></p>
    </div>
    <table>
        <tr>
            <th>Test Name</th>
            <th>Score</th>
            <th>Status</th>
            <th>Metrics</th>
        </tr>
"""
        for result in results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            status_class = "passed" if result.passed else "failed"
            metrics_str = ", ".join(f"{k}: {v:.2f}" for k, v in result.metrics.items())
            html += f"""        <tr>
            <td>{result.test_name}</td>
            <td class="score">{result.score:.2f}</td>
            <td class="{status_class}">{status}</td>
            <td>{metrics_str}</td>
        </tr>
"""
        
        html += """    </table>
</body>
</html>"""
        return html
    
    def save(self, results: List[EvaluationResult], path: str):
        """Save HTML report to file."""
        content = self.generate(results)
        Path(path).write_text(content)
        print(f"HTML report saved to: {path}")


class SlackReporter(BaseReporter):
    """Send reports to Slack."""
    
    def __init__(self, webhook_url: str):
        """
        Initialize Slack reporter.
        
        Args:
            webhook_url: Slack webhook URL
        """
        self.webhook_url = webhook_url
    
    def generate(self, results: List[EvaluationResult]) -> str:
        """Generate Slack message."""
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        avg_score = sum(r.score for r in results) / total if total else 0
        
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "🧪 AI Testing Report"}
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Total Tests:*\n{total}"},
                    {"type": "mrkdwn", "text": f"*Passed:*\n{passed}"},
                    {"type": "mrkdwn", "text": f"*Failed:*\n{failed}"},
                    {"type": "mrkdwn", "text": f"*Avg Score:*\n{avg_score:.2%}"},
                ]
            }
        ]
        return json.dumps(blocks)
    
    def send(self, results: List[EvaluationResult], mention: str = None):
        """Send report to Slack."""
        import requests
        
        message = self.generate(results)
        if mention:
            message = message.replace('"text":', f'"text": "{mention} ",')
        
        try:
            response = requests.post(
                self.webhook_url,
                data=message.encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            print("Slack notification sent!")
        except Exception as e:
            print(f"Failed to send Slack notification: {e}")
    
    def save(self, results: List[EvaluationResult], path: str):
        """Save JSON representation (for compatibility)."""
        content = self.generate(results)
        Path(path).write_text(content)


class ConsoleReporter(BaseReporter):
    """Print results to console."""
    
    def generate(self, results: List[EvaluationResult]) -> str:
        """Generate console output."""
        output = ["\n" + "="*50]
        output.append("🧪 AI TESTING RESULTS")
        output.append("="*50 + "\n")
        
        for result in results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            output.append(f"{result.test_name}: {status} (Score: {result.score:.2f})")
            for metric, score in result.metrics.items():
                output.append(f"  - {metric}: {score:.2f}")
            output.append("")
        
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        avg = sum(r.score for r in results) / total if total else 0
        
        output.append("-"*50)
        output.append(f"Total: {total} | Passed: {passed} | Avg Score: {avg:.2%}")
        
        return "\n".join(output)
    
    def save(self, results: List[EvaluationResult], path: str = None):
        """Print to console."""
        print(self.generate(results))


__all__ = [
    "BaseReporter",
    "JSONReporter", 
    "HTMLReporter",
    "SlackReporter",
    "ConsoleReporter",
]
