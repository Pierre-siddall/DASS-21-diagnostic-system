import main
import unittest


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




if __name__ == "__main__":
    unittest.main()
