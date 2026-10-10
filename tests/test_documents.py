import unittest

from app.documents import chunk_text, extract_text


class DocumentTests(unittest.TestCase):
    def test_extracts_utf8_text_and_markdown(self):
        self.assertEqual(extract_text("notes.txt", b"\xef\xbb\xbfhello"), "hello")
        self.assertEqual(extract_text("notes.md", b"# Heading"), "# Heading")

    def test_rejects_unsupported_file_types(self):
        with self.assertRaisesRegex(ValueError, "Unsupported file type"):
            extract_text("image.png", b"not text")

    def test_chunks_text_with_overlap_and_preserves_content(self):
        text = "alpha beta gamma delta epsilon zeta eta theta"
        chunks = chunk_text(text, chunk_size=18, overlap=5)

        self.assertGreater(len(chunks), 1)
        self.assertEqual([chunk.index for chunk in chunks], list(range(len(chunks))))
        self.assertTrue(all(len(chunk.text) <= 18 for chunk in chunks))
        for word in text.split():
            self.assertTrue(any(word in chunk.text for chunk in chunks), word)

    def test_empty_text_returns_no_chunks(self):
        self.assertEqual(chunk_text(" \n\t ", chunk_size=20, overlap=0), [])

    def test_invalid_chunk_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            chunk_text("text", chunk_size=10, overlap=10)


if __name__ == "__main__":
    unittest.main()
