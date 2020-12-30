"""Unit test for Shakespere dataset"""

from absl.testing import absltest
from absl.testing import flagsaver
# see, https://github.com/abseil/abseil-py/blob/master/absl/testing/flagsaver.py
from absl.testing import parameterized
# see, https://github.com/abseil/abseil-py/blob/master/absl/testing/parameterized.py

from sentiment_classifier.dataset import shakespeare


class TestShakespeareDataset(absltest.TestCase):
    """Test for """

    # pylint: disable=protected-access
    def setUp(self):
        pass

    def test_make_dictionary(self):
        """Test for _make_dictionary"""
        test_text = 'hello'
        dictionaries = shakespeare._make_dictionary(test_text)
        expected_dict = {'e': 0, 'h': 1, 'l': 2, 'o': 3}
        expected_list = expected_dict.keys()
        self.assertEqual(dictionaries[0], expected_dict)
        self.assertTrue(
            all([a == b for a, b in zip(dictionaries[1], expected_list)]))

    def test_encode(self):
        """Test for _encode"""
        test_text = 'hello'
        dictionaries = shakespeare._make_dictionary(test_text)
        encoded = shakespeare._encode(test_text, dictionaries[0])
        expected_result = [1, 0, 2, 2, 3]
        self.assertTrue(all([a == b
                             for a, b in zip(encoded, expected_result)]))

    def test_get_dataset(self):
        """Test for get_dataset"""
        shakespeare.get_dataset()
        self.assertTrue(True)

    def tearDown(self):
        pass


if __name__ == '__main__':
    absltest.main()
