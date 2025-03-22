import unittest
from Models.Patcher import TestPatchEmbedding  # Replace 'your_module' with your actual module name
from Models.MSA import TestMultiHeadSelfAttentionBlock
from Models.MLP import TestMultipLayerPerceptron
from Models.TransformerEncoder import TestTransformerEncoder
from Models.ViT import TestViT


# Discover and run all test cases in the current directory
def run_tests():
    test_suite = unittest.TestSuite()
    test_loader = unittest.TestLoader()


    test_suite.addTest(test_loader.loadTestsFromTestCase(TestPatchEmbedding))
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestMultiHeadSelfAttentionBlock))
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestMultipLayerPerceptron))
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestTransformerEncoder))
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestViT))
    # Run the test suite
    runner = unittest.TextTestRunner()
    runner.run(test_suite)

if __name__ == '__main__':
    run_tests()
