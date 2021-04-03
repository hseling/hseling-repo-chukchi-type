import unittest

import hseling_api_chukchi_type


class HSELing_API_Chukchi_typeTestCase(unittest.TestCase):

    def setUp(self):
        self.app = hseling_api_chukchi_type.app.test_client()

    def test_index(self):
        rv = self.app.get('/healthz')
        self.assertIn('Application Chukchi Type', rv.data.decode())


if __name__ == '__main__':
    unittest.main()
