import unittest
from run_predict import *
from inverse_normalize import INVERSE_NORMALIZERS

inverse_normalizer = INVERSE_NORMALIZERS['nemo']
hindi_digits_with_zero = '0123456789'


class HindiInverseTextNormalization(unittest.TestCase):

    def test_two_digit_numbers_are_converted_to_numerals(self):

        data = ['रीटा के पास सोलह बिल्लियाँ हैं।']
        expected_output = ['रीटा के पास 16 बिल्लियाँ हैं।']

        inverse_normalizer_prediction = inverse_normalizer(data, verbose=False)

        astr_list = []
        inverse_normalizer_prediction = [sent.replace('\r', '') for sent in inverse_normalizer_prediction]
        for sent in inverse_normalizer_prediction:
            trimmed_sent = ' '.join(
                [remove_starting_zeros(word, hindi_digits_with_zero) for word in sent.split(' ')])
            astr_list.append(trimmed_sent)

        self.assertEqual(expected_output, astr_list)

    def test_hundreds_are_converted_to_numerals(self):

        data = ['रीटा के पास चार सौ बीस बिल्लियाँ हैं।']
        expected_output = ['रीटा के पास 420 बिल्लियाँ हैं।']

        inverse_normalizer_prediction = inverse_normalizer(data, verbose=False)

        astr_list = []
        inverse_normalizer_prediction = [sent.replace('\r', '') for sent in inverse_normalizer_prediction]
        for sent in inverse_normalizer_prediction:
            trimmed_sent = ' '.join(
                [remove_starting_zeros(word, hindi_digits_with_zero) for word in sent.split(' ')])
            astr_list.append(trimmed_sent)

        self.assertEqual(expected_output, astr_list)

    def test_thousands_are_converted_to_unformatted_numerals(self):
        # no formatting in indian format for this test
        data = ['एक हज़ार चार सौ बीस', 'बारह हज़ार सात सौ तीन', 'पंद्रह सौ', 'पंद्रह सौ सात']
        expected_output = ['1420', '12703', '1500', '1507']

        inverse_normalizer_prediction = inverse_normalizer(data, verbose=False)

        astr_list = []
        inverse_normalizer_prediction = [sent.replace('\r', '') for sent in inverse_normalizer_prediction]
        for sent in inverse_normalizer_prediction:
            trimmed_sent = ' '.join(
                [remove_starting_zeros(word, hindi_digits_with_zero) for word in sent.split(' ')])
            astr_list.append(trimmed_sent)

        self.assertEqual(expected_output, astr_list)

    def test_lakhs_are_converted_to_unformatted_numerals(self):
        # no formatting in indian format for this test
        # TODO: 'दो लाख पंद्रह सौ'
        data = ['दो लाख', 'दो लाख चार सौ', 'चार लाख चार सौ चार', 'बारह लाख बीस हज़ार सात सौ पंद्रह']
        expected_output = ['200000', '200400', '400404', '1220715']

        inverse_normalizer_prediction = inverse_normalizer(data, verbose=False)

        astr_list = []
        inverse_normalizer_prediction = [sent.replace('\r', '') for sent in inverse_normalizer_prediction]
        for sent in inverse_normalizer_prediction:
            trimmed_sent = ' '.join(
                [remove_starting_zeros(word, hindi_digits_with_zero) for word in sent.split(' ')])
            astr_list.append(trimmed_sent)

        self.assertEqual(expected_output, astr_list)

    def test_thousands_and_lakhs_are_converted_to_formatted_numerals(self):

        data = ['एक हज़ार चार सौ बीस', 'बारह हज़ार सात सौ तीन', 'चार लाख चार सौ चार',
                'बारह लाख बीस हज़ार सात सौ पंद्रह']
        expected_output = ['1,420', '12,703', '4,00,404', '12,20,715']

        inverse_normalizer_prediction = inverse_normalizer(data, verbose=False)

        astr_list = []
        comma_sep_num_list = []
        inverse_normalizer_prediction = [sent.replace('\r', '') for sent in inverse_normalizer_prediction]
        for sent in inverse_normalizer_prediction:
            trimmed_sent = ' '.join(
                [remove_starting_zeros(word, hindi_digits_with_zero) for word in sent.split(' ')])
            astr_list.append(trimmed_sent)
            comma_sep_num_list.append(
                ' '.join([indian_format(word, hindi_digits_with_zero) for word in trimmed_sent.split(' ')]))

        self.assertEqual(expected_output, comma_sep_num_list)

    def test_single_and_double_digit_crores_are_converted_to_formatted_numerals(self):
        data = ['चार करोड़ इक्कीस लाख', 'चार करोड़ इक्कीस लाख चार हज़ार चार सौ चार',
                'बत्तीस करोड़ इक्कीस लाख सैंतीस हज़ार चार सौ बारह', 'बत्तीस करोड़ चार सौ']
        expected_output = ['4,21,00,000', '4,21,04,404', '32,21,37,412', '32,00,00,400']

        inverse_normalizer_prediction = inverse_normalizer(data, verbose=False)

        astr_list = []
        comma_sep_num_list = []
        inverse_normalizer_prediction = [sent.replace('\r', '') for sent in inverse_normalizer_prediction]
        for sent in inverse_normalizer_prediction:
            trimmed_sent = ' '.join(
                [remove_starting_zeros(word, hindi_digits_with_zero) for word in sent.split(' ')])
            astr_list.append(trimmed_sent)
            comma_sep_num_list.append(
                ' '.join([indian_format(word, hindi_digits_with_zero) for word in trimmed_sent.split(' ')]))

        self.assertEqual(expected_output, comma_sep_num_list)

    def test_spoken_form_of_single_digit_thousands_for_years_are_converted(self):
        # TODO: don't format (comma) for years
        data = ['वर्ष उन्निस सौ चौहत्तर', 'लेखों की संख्या एक हज़ार नौ सौ चौहत्तर हैं।']
        expected_output = ['वर्ष 1,974', 'लेखों की संख्या 1,974 हैं।']

        inverse_normalizer_prediction = inverse_normalizer(data, verbose=False)

        astr_list = []
        comma_sep_num_list = []
        inverse_normalizer_prediction = [sent.replace('\r', '') for sent in inverse_normalizer_prediction]
        for sent in inverse_normalizer_prediction:
            trimmed_sent = ' '.join(
                [remove_starting_zeros(word, hindi_digits_with_zero) for word in sent.split(' ')])
            astr_list.append(trimmed_sent)
            comma_sep_num_list.append(
                ' '.join([indian_format(word, hindi_digits_with_zero) for word in trimmed_sent.split(' ')]))

        self.assertEqual(expected_output, comma_sep_num_list)


if __name__ == '__main__':
    unittest.main()
