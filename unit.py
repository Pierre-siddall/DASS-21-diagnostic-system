import main
import unittest
import spacy


class TestTextExtraction(unittest.TestCase):
    def test_datatype(self):
        data = main.extract_training_text("sectraining.csv")
        self.assertEqual(type(data[0]), list)
        self.assertEqual(type(data[0][0]), str)

    def test_empty(self):
        data = main.extract_training_text("sectraining.csv")

        if len(data) != 0:
            all_result = False
        else:
            all_result = True

        if len(data[0]) != 0:
            individual_result = False
        else:
            individual_result = True

        self.assertFalse(all_result)
        self.assertFalse(individual_result)

    def test_number_removal(self):
        data = main.extract_training_text("sectraining.csv")

        removal = True

        for text in data:
            if len(text) != 1:
                removal = False

        self.assertTrue(removal)


class TestFilterTags(unittest.TestCase):
    def test_datatype(self):
        nlp = spacy.load("en_core_web_sm")
        data = main.extract_training_text("sectraining.csv")
        test_case = main.tokenize_text(nlp, data[0][0])
        filtered_test_case = main.filter_tags(test_case)

        self.assertEqual(type(filtered_test_case), list)
        self.assertEqual(type(filtered_test_case[0]), spacy.tokens.token.Token)

    def test_empty(self):

        nlp = spacy.load("en_core_web_sm")
        data = main.extract_training_text("sectraining.csv")
        test_case = main.tokenize_text(nlp, data[0][0])
        filtered_test_case = main.filter_tags(test_case)

        if len(filtered_test_case) != 0:
            result = False
        else:
            result = True

        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
