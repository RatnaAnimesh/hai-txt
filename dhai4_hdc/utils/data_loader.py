import re

class CorpusLoader:
    def __init__(self, file_path=None):
        """
        Simple text generator for streaming corpora (like TinyStories) 
        token by token without loading massive files into memory.
        """
        self.file_path = file_path
        
    def stream_tokens(self, text_content=None):
        """
        Yields normalized tokens one by one.
        Can process a direct string (for testing) or a file.
        """
        if text_content:
            yield from self._tokenize(text_content)
        elif self.file_path:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    yield from self._tokenize(line)
        else:
            raise ValueError("Must provide file_path or text_content")

    def _tokenize(self, text: str):
        # Extremely basic tokenizer: lowercase, strip punctuation except basic sentence ends
        text = text.lower()
        # Keep basic punctuation for structural cues if needed, but strip weird chars
        words = re.findall(r'\b\w+\b|[.!?]', text)
        for w in words:
            yield w
