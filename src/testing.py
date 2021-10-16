import unittest
from gui_model import Model


class MyTestCase(unittest.TestCase):
    m = Model()

    def test_preprocessing(self):
        m = Model()
        preprocessed = "yeah   definitely annoying!"
        sentence = "Yeah, it's definitely annoying!"
        pre = str(m.preprocess_text(sentence))
        print(type(pre))
        print(pre)
        print(type(preprocessed))
        assert(pre == preprocessed)




if __name__ == '__main__':
    unittest.main()
