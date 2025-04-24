import unittest
import re

# Import the preprocess_text function
# Assuming the original code is in a file named emotion_datasets.py
from data_processing import preprocess_text

class TestPreprocessText(unittest.TestCase):
    """Test the preprocess_text function"""
    
    def test_lowercase_conversion(self):
        """Test that text is converted to lowercase"""
        text = "This Is A Test"
        processed = preprocess_text(text)
        self.assertEqual(processed, "this is a test")
    
    def test_url_removal(self):
        """Test that URLs are removed"""
        text = "Check this link https://example.com and this www.test.org"
        processed = preprocess_text(text)
        self.assertEqual(processed, "check this link and this")
    
    def test_mention_removal(self):
        """Test that @mentions are removed"""
        text = "Hello @username how are you?"
        processed = preprocess_text(text)
        self.assertEqual(processed, "hello how are you?")
    
    def test_hashtag_processing(self):
        """Test that hashtags are properly processed (# removed, word kept)"""
        text = "I love #Python and #coding"
        processed = preprocess_text(text)
        self.assertEqual(processed, "i love python and coding")
    
    def test_retweet_removal(self):
        """Test that RT indicators are removed"""
        text = "RT This is a retweet"
        processed = preprocess_text(text)
        self.assertEqual(processed, "this is a retweet")
    
    def test_special_char_removal(self):
        """Test that special characters are removed while keeping basic punctuation"""
        text = "Hey! This has $special^ characters."
        processed = preprocess_text(text)
        self.assertEqual(processed, "hey! this has special characters.")
    
    def test_whitespace_normalization(self):
        """Test that extra whitespace is normalized"""
        text = "  Too   many    spaces   "
        processed = preprocess_text(text)
        self.assertEqual(processed, "too many spaces")
    
    def test_combined_preprocessing(self):
        """Test all preprocessing steps together"""
        text = "RT @user I #loved this https://example.com!!! It's $great^"
        processed = preprocess_text(text)
        self.assertEqual(processed, "i loved this its great")
    
    def test_punctuation_preservation(self):
        """Test that basic punctuation is preserved"""
        text = "Hello! How are you? This is a test."
        processed = preprocess_text(text)
        self.assertEqual(processed, "hello! how are you? this is a test.")
    
    def test_emoji_handling(self):
        """Test handling of emoji characters"""
        text = "I love this 😍 so much!"
        processed = preprocess_text(text)
        # Special characters should be removed, but text preserved
        self.assertEqual(processed, "i love this so much!")
    
    def test_empty_string(self):
        """Test processing an empty string"""
        text = ""
        processed = preprocess_text(text)
        self.assertEqual(processed, "")
    
    def test_only_special_chars(self):
        """Test processing text that contains only special characters"""
        text = "@#$%^&*"
        processed = preprocess_text(text)
        self.assertEqual(processed, "")


# For stand-alone testing, add this
if __name__ == '__main__':
    unittest.main()