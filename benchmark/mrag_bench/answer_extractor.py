#!/usr/bin/env python3
"""Extract multiple-choice answers (A/B/C/D) from VLM responses."""

import re
from typing import Optional


# Patterns ordered by specificity (most specific first)
ANSWER_PATTERNS = [
    # Explicit answer statements
    r"(?:the\s+)?(?:correct\s+)?answer\s+is\s*[:\s]*\(?([A-Da-d])\)?",
    r"(?:the\s+)?(?:best\s+)?(?:correct\s+)?(?:answer|option|choice)\s*[:\s]*\(?([A-Da-d])\)?",
    r"I(?:'d|'ll|\s+would)\s+(?:choose|select|pick|go\s+with)\s+\(?([A-Da-d])\)?",
    r"(?:my\s+)?(?:answer|choice|selection)\s+(?:is|would\s+be)\s*[:\s]*\(?([A-Da-d])\)?",

    # Option format patterns
    r"\b(?:option|choice)\s+\(?([A-Da-d])\)?(?:\s+is\s+correct)?",
    r"\(?([A-Da-d])\)?\s+is\s+(?:the\s+)?(?:correct|right|best)\s+(?:answer|option|choice)?",

    # Simple letter patterns (with context)
    r"^([A-Da-d])\s*[:\.\)]",  # Letter at start of line with delimiter
    r"\n([A-Da-d])\s*[:\.\)]",  # Letter at start of new line
    r"(?:select|chose|pick|choose)\s+\(?([A-Da-d])\)?",

    # Bracketed/parenthesized standalone
    r"\[([A-Da-d])\]",
    r"\(([A-Da-d])\)",

    # Final fallback: standalone letter (last resort)
    r"(?:^|\s)([A-Da-d])(?:\s*$|\s*\.?\s*$)",
]


def extract_answer(response: str) -> Optional[str]:
    """Extract a single-letter answer (A, B, C, or D) from VLM response.

    Uses a series of regex patterns to identify the most likely intended answer
    from various response formats.

    Args:
        response: The VLM's text response to analyze.

    Returns:
        A single uppercase letter (A, B, C, or D) if found, else None.

    Examples:
        >>> extract_answer("The answer is B.")
        'B'
        >>> extract_answer("I would choose option C because...")
        'C'
        >>> extract_answer("A) This is correct")
        'A'
        >>> extract_answer("Based on the image, D is the best choice.")
        'D'
    """
    if not response:
        return None

    # Normalize whitespace
    text = response.strip()

    # Try each pattern in order
    for pattern in ANSWER_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            letter = match.group(1).upper()
            if letter in "ABCD":
                return letter

    # If no pattern matched, try to find any standalone A/B/C/D
    # Only use this if the response is very short (likely just the answer)
    if len(text) <= 10:
        for char in text.upper():
            if char in "ABCD":
                return char

    return None


def extract_answer_with_confidence(response: str) -> tuple[Optional[str], float]:
    """Extract answer with confidence score.

    Args:
        response: The VLM's text response.

    Returns:
        Tuple of (answer, confidence) where confidence is 0.0-1.0.
    """
    if not response:
        return None, 0.0

    text = response.strip()

    # High confidence patterns (explicit answer statements)
    high_confidence_patterns = ANSWER_PATTERNS[:4]
    for pattern in high_confidence_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            letter = match.group(1).upper()
            if letter in "ABCD":
                return letter, 0.95

    # Medium confidence patterns
    medium_confidence_patterns = ANSWER_PATTERNS[4:8]
    for pattern in medium_confidence_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            letter = match.group(1).upper()
            if letter in "ABCD":
                return letter, 0.75

    # Low confidence patterns
    for pattern in ANSWER_PATTERNS[8:]:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            letter = match.group(1).upper()
            if letter in "ABCD":
                return letter, 0.5

    # Very low confidence: short response fallback
    if len(text) <= 10:
        for char in text.upper():
            if char in "ABCD":
                return char, 0.3

    return None, 0.0


def test_extractor():
    """Test the answer extractor with various response formats."""
    test_cases = [
        ("The answer is B.", "B"),
        ("I would choose option C because it matches the image.", "C"),
        ("A) This is the correct answer", "A"),
        ("Based on the image, D is the best choice.", "D"),
        ("The correct answer is (A).", "A"),
        ("B", "B"),
        ("After analyzing the image, I believe the answer is C.", "C"),
        ("Option A is correct.", "A"),
        ("I'll go with D.", "D"),
        ("[B]", "B"),
        ("My choice is A.", "A"),
        ("This is clearly B because...", "B"),
        ("", None),
        ("I don't know the answer.", None),
    ]

    print("Testing answer extractor:")
    print("-" * 60)

    passed = 0
    for response, expected in test_cases:
        result = extract_answer(response)
        status = "PASS" if result == expected else "FAIL"
        if result == expected:
            passed += 1
        print(f"{status}: '{response[:40]}...' -> {result} (expected {expected})")

    print("-" * 60)
    print(f"Passed: {passed}/{len(test_cases)}")


if __name__ == "__main__":
    test_extractor()
