import unittest
# import sys
# sys.path.append('../')
from run_predict import *
from inverse_normalize import INVERSE_NORMALIZERS


class HindiInverseTextNormalization(unittest.TestCase):
    def test_two_digit_numbers_are_conveted_to_numerals(self):
        inverse_normalizer = 'nemo'
        inverse_normalizer = INVERSE_NORMALIZERS[inverse_normalizer]

        data = ['रीटा के पास सोलह बिल्लियाँ हैं।']
        expected_output = ['रीटा के पास 16 बिल्लियाँ हैं।']
        hindi_digits_with_zero = '0123456789'

        inverse_normalizer_prediction = inverse_normalizer(data, verbose=False)

        astr_list = []
        inverse_normalizer_prediction = [sent.replace('\r', '') for sent in inverse_normalizer_prediction]
        for sent in inverse_normalizer_prediction:
            trimmed_sent = ' '.join([remove_starting_zeros(word, hindi_digits_with_zero) for word in sent.split(' ')])
            astr_list.append(trimmed_sent)

        self.assertEqual(expected_output, astr_list)


if __name__ == '__main__':
    unittest.main()
