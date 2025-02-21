import re
import json
from pathlib import Path
from turtle import color
from spellchecker import SpellChecker
from string import punctuation
from typing import List, Set
from termcolor import colored


class RSTSpellChecker:
    def __init__(self, spell_checker: SpellChecker):

        self.spell_checker = spell_checker
        self.sphinx_block = False

        # Word regex, these are ran for every word so compile once up here.
        # Numeric regex
        # https://stackoverflow.com/questions/1323364/in-python-how-to-check-if-a-string-only-contains-certain-characters
        # [Prefix syms][nums][intermediate syms][trailing nums][end syms]
        self.re_numeric = re.compile(r"^[+\-(vx]*[0-9]+[+\-xe \.]*[0-9]*[xDk%\.]*$")
        # Left over sphinx keys
        self.re_sphinx_keys = re.compile(r"\s*(:alt:)\s*")
        # Stuff from python code blocks and url stuff
        self.re_code_words = re.compile(
            r"(.*\.py|.*\.html|.*\.org|.*\.com|.*\.vti|.*\.vtu|.*\.vtp)"
        )
        # All caps for abbrv (can have trailing s)
        self.re_caps = re.compile(r"^[^a-z]*[s]?$")

    def check_sphinx_block(self, line: str) -> bool:
        """Determins if line is in a code, math or table block based on indent whitespace

        Parameters
        ----------
        line : str
            line of text

        Returns
        -------
        bool
            If line is in code block
        """
        # code block
        re_sphinx_code_block = re.compile(r"^\s*\.\.\s+(code::|code-block::)")
        # math or table block
        re_sphinx_math_block = re.compile(r"^\s*\.\.\s+(math::|table::)")
        # Leading white space check
        re_white_space = re.compile(
            r"^(\s{2,}|\t+)"
        )  # Assuming tab spacing has at least 2 spaces

        # Check for start of code or math block
        if bool(re_sphinx_code_block.search(line)):
            self.sphinx_block = True
            return self.sphinx_block
        elif bool(re_sphinx_math_block.search(line)):
            self.sphinx_block = True
            return self.sphinx_block
        # Else check to see if exempt line or non-indendent line
        if self.sphinx_block:
            # End of code block is a line with no white space at the start (no-indent)
            if (
                not bool(re_white_space.search(line))
                and len(re.sub("[\s+]", "", line)) > 0
            ):
                self.sphinx_block = False

        return self.sphinx_block

    def exempt_lines(self, line: str) -> bool:
        """Checks if line should be exempt from checking, this applys for various
        sphinx sections such as code blocks, figures, tables, etc.

        Parameters
        ----------
        line : str
            line of text

        Returns
        -------
        bool
            If line should be skipped
        """
        re_sphinx_code_ref = re.compile(
            r"code::|role::|literalinclude:|:language:|:lines:|:format:|:start-after:|:end-before:"
        )
        re_sphinx_fig_ref = re.compile(
            r"(^..\s*figure::|^\s*:width:|^\s*:align:|^\s*:name:|^\s*:header-rows:)"
        )
        re_title_boaders = re.compile(r"^=+\s+$|^~+\s+$|^\^+\s+$")
        re_sphinx_citation = re.compile(r"^\s*\.\. \[#.*\]")
        re_sphinx_ref_target = re.compile(r"^\s*\.\.\s+\_.*:\s*$")
        re_sphinx_math = re.compile(r"^\s*\.\.\s+math::")

        if bool(re_sphinx_code_ref.search(line)):
            return True
        elif bool(re_sphinx_fig_ref.search(line)):
            return True
        elif bool(re_title_boaders.search(line)):
            return True
        elif bool(re_sphinx_citation.search(line)):
            return True
        elif bool(re_sphinx_ref_target.search(line)):
            return True
        elif bool(re_sphinx_math.search(line)):
            return True

        return False

    def exempt_word(self, word: str) -> bool:
        """Checks for words that should be exempt from spell checking

        Parameters
        ----------
        word : str
            Word string

        Returns
        -------
        bool
            If work should be exempt
        """
        # Numericals (numbers, #-#, #x#)
        if bool(self.re_numeric.search(word)):
            return True
        if bool(self.re_sphinx_keys.search(word)):
            return True
        if bool(self.re_code_words.search(word)):
            return True
        # All cap abbrive
        if bool(self.re_caps.search(word)):
            return True
        # Works with back-slashes (escape characters, aka weird stuff)
        if "\\" in word:
            return True

        return False

    def prepare_line(self, line: str) -> List[str]:
        """Prepares test line for parsing, will check if line should be skipped,
        remove any sphinx keywords, then split into words based on white space.

        Parameters
        ----------
        line : str
            Line of text

        Returns
        -------
        List[str]
            List of keywords
        """
        # Check if line is in sphinx block or is an exempt line
        if self.check_sphinx_block(line):
            return []
        if self.exempt_lines(line):
            return []
        # Remove specifc parts of the line that are sphinx items
        re_sphinx_inline = re.compile(r"(:ref:|:math:|:numref:|:eq:|:code:)`.*?`")
        re_sphinx_code = re.compile(r"(``.*?``|`.*?`)")
        re_sphinx_cite = re.compile(r"\[#.*?\]\_")
        re_sphinx_link = re.compile(r"<.*?>`\_")
        re_sphinx_block_titles = re.compile(
            r"(\.\.\s+table::|\.\.\s+list-table::|\.\.\s+note::)"
        )

        line = line.strip("\n")

        if bool(re_sphinx_inline.search(line)):
            line = re_sphinx_inline.sub(r"", line)
        if bool(re_sphinx_code.search(line)):
            line = re_sphinx_code.sub(r"", line)
        if bool(re_sphinx_cite.search(line)):
            line = re_sphinx_cite.sub(r"", line)
        if bool(re_sphinx_link.search(line)):
            line = re_sphinx_link.sub(r"", line)
        if bool(re_sphinx_block_titles.search(line)):
            line = re_sphinx_block_titles.sub(r"", line)

        # Split up sentence into words
        words = re.split(r"(\s+|/)", line)

        # Filter empty strings
        words = list(filter(None, words))

        return words

    def get_unknown_words(self, line: str) -> List[str]:
        """Gets unknown words not present in spelling dictionary

        Parameters
        ----------
        line : str
            Line of text to parse

        Returns
        -------
        List[str]
            List of unknown words (if any)
        """
        # Clean line and split into list of words
        words = self.prepare_line(line)
        # Primative plural word checking
        re_plural = re.compile(r"(\’s|\'s|s\'|s\’|s|\(s\))$")
        unknown_words = []
        for word0 in words:
            # Check for miss-spelling of word and without trailing s
            if word0 in self.spell_checker or self.exempt_word(word0):
                continue
            # Strip punctuation and check again
            word = word0.strip(punctuation)
            if word in self.spell_checker or self.exempt_word(word):
                continue
            # Add dot after stripping punctuation for abbrv
            word = word0.strip(punctuation) + "."
            if word in self.spell_checker or self.exempt_word(word):
                continue
            # Strip plural / posessive
            word = re_plural.sub(r"", word0)
            if word in self.spell_checker or self.exempt_word(word):
                continue
            # Strip plural after punctuation
            word = re_plural.sub(r"", word0.strip(punctuation))
            if word in self.spell_checker or self.exempt_word(word):
                continue
            # If none of these combos worked mark as unknown
            unknown_words.append(word0.strip(punctuation))

        return unknown_words


def test_rst_spelling(
    userguide_path: Path,
    en_dictionary_path: Path = Path("./test/en_dictionary.json.gz"),
    extra_dictionary_path: Path = Path("./test/physicsnemo_dictionary.json"),
    file_pattern: str = "*.rst",
):
    """Looks through RST files for any references to example python files

    Parameters
    ----------
    userguide_path : Path
        Path to user guide RST files
    en_dictionary_path: Path, optional
        Path to english dictionary
    extra_dictionary_path: Path, optional
        Path to additional PhysicsNeMo dictionary
    file_pattern : str, optional
        Pattern for file types to parse, by default "*.rst"

    Raises
    -------
    ValueError: If spelling errors have been found
    """
    assert userguide_path.is_dir(), "Invalid user guide folder path"
    assert en_dictionary_path.is_file(), "Invalid english dictionary path"
    assert extra_dictionary_path.is_file(), "Invalid additional dictionary path"

    spell = SpellChecker(language=None, distance=2)
    spell.word_frequency.load_dictionary(str(en_dictionary_path), encoding="utf-8")
    # Can be used to export current dictionary for merging dicts
    # spell.export('en_dictionary.json.gz', gzipped=True)
    # Load extra words
    data = json.load(open(extra_dictionary_path))
    spell.word_frequency.load_words(data["dictionary"])
    rst_checker = RSTSpellChecker(spell)

    spelling_errors = []
    spelling_warnings = []
    for doc_file in userguide_path.rglob(file_pattern):
        for i, line in enumerate(open(doc_file)):
            # Clean line and split into list of words
            words = rst_checker.get_unknown_words(line)
            for word in words:
                # Get the most likely correction
                corr_word = spell.correction(word)
                # If there is a potential correct work in dictionary flag as error
                if not corr_word == word:
                    err_msg = (
                        f'Found potential spelling error: "{word.lower()}", did you mean "{corr_word}"?'
                        + "\n"
                    )
                    err_msg += (
                        f"Located in File: {doc_file}, Line: {i}, Word: {word}" + "\n"
                    )
                    spelling_errors.append(colored(err_msg, "red"))
                # Otherwise make it a warning as a unrecognizable word
                else:
                    err_msg = (
                        f"Unknown word: {word}, consider adding to dictionary." + "\n"
                    )
                    err_msg += (
                        f"Located in File: {doc_file}, Line: {i}, Word: {word}" + "\n"
                    )
                    spelling_warnings.append(colored(err_msg, "yellow"))

    # Print warnings
    if len(spelling_warnings) > 0:
        print(colored("Spelling WARNINGS:", "yellow"))
        for msg in spelling_warnings:
            print(msg)
    # Print likely spelling errors
    if len(spelling_errors) > 0:
        print(colored("Spelling ERRORS:", "red"))
        for msg in spelling_errors:
            print(msg)

    if len(spelling_errors) > 0:
        raise ValueError(
            "Spelling errors found, either correct or add new words to dictionary."
        )


if __name__ == "__main__":
    # Paths inside CI docker container
    user_guide_path = Path("./user_guide")
    test_rst_spelling(user_guide_path)
