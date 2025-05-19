from abc import abstractmethod


class Formator:
    @abstractmethod
    def format_input(self, text: str):
        pass

    def __str__(self) -> str:
        return self.__class__.__name__


class SimplerFormator(Formator):
    def format_input(self, text: str):
        return f"Please rewrite the following text in a way that is simpler and easier to understand, using clear language and shorter sentences without losing the original meaning:\n\n{text}"


class InformalTransfer(Formator):
    def format_input(self, text: str):
        return f"Transform the following text into an informal style:\n\n{text}"


class FormalTransfer(Formator):
    def format_input(self, text: str):
        return f"Rewrite the following text in a formal style:\n\n{text}"


class EnZhTransfer(Formator):
    def format_input(self, text: str):
        return f"Please translate the following English text into Chinese:\n\n{text}"


class ZhEnTransfer(Formator):
    def format_input(self, text: str):
        return f"Please translate the following Chinese text into English:\n\n{text}"


class RephraseFormator(Formator):
    def format_input(self, text: str):
        return f"please rephrase the text below:\n\n{text}"


class RewriteFormator(Formator):
    def format_input(self, text: str):
        return f"please rewrite the following text:\n\n{text}"


class PolishFormator(Formator):
    def format_input(self, text: str):
        return f"please polish the following text:\n\n{text}"


class ParaphraseFormator(Formator):
    def format_input(self, text: str):
        return f"please paraphrase the following text:\n\n{text}"
