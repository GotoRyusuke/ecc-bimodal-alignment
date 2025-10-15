from dataclasses import dataclass

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float

@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start

@dataclass
class WordStamp:
    label: str
    start: float
    end: float

    def __repr__(self):
        return f"{self.label}: {self.start:5.2f}-{self.end:5.2f}sec"

    @property
    def length(self):
        return self.end - self.start






