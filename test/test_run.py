import unittest
import xnmt.xnmt_run_experiments as run

class TestRunningConfig(unittest.TestCase):
  def test_debug_yaml(self):
    run.main(["test/config/translator.yaml"])

  def test_debug_report(self):
    run.main(["test/config/translator_report.yaml"])

  # TODO: fix tests
#  def test_debug_segmenting(self):
#    run.main(["test/config/segmenting.yaml"])
#  def test_debug_segmentin2g(self):
#    run.main(["test/config/segmenting2.yaml"])
#  def test_debug_segmentin2g(self):
#    run.main(["test/config/retrieval.yaml"])

if __name__ == "__main__":
  unittest.main()
