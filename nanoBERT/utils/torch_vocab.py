from collections import OrderedDict
from typing import Dict, Iterable, List, Mapping, Optional


class Vocab:
    """Small Vocab-compatible token/index mapping.

    The project only needs vocabulary bookkeeping. Keeping this implementation
    local avoids external binary dependencies on specific
    PyTorch releases.
    """

    def __init__(self, tokens: Iterable[str] = (), default_index: Optional[int] = None) -> None:
        self.itos_: List[str] = []
        self.stoi_: Dict[str, int] = {}
        self.default_index_ = default_index

        for token in tokens:
            self.append_token(token)

    @property
    def vocab(self) -> "Vocab":
        """Compatibility with wrappers that expose `.vocab`."""
        return self

    @property
    def is_jitable(self) -> bool:
        return False

    def forward(self, tokens: List[str]) -> List[int]:
        return self.lookup_indices(tokens)

    def __call__(self, tokens: List[str]) -> List[int]:
        return self.lookup_indices(tokens)

    def __len__(self) -> int:
        return len(self.itos_)

    def __contains__(self, token: str) -> bool:
        return token in self.stoi_

    def __getitem__(self, token: str) -> int:
        if token in self.stoi_:
            return self.stoi_[token]
        if self.default_index_ is not None:
            return self.default_index_
        raise KeyError(f"Token {token!r} not found and default index is not set.")

    def set_default_index(self, index: Optional[int]) -> None:
        self.default_index_ = index

    def get_default_index(self) -> Optional[int]:
        return self.default_index_

    def insert_token(self, token: str, index: int) -> None:
        if token in self.stoi_:
            raise RuntimeError(f"Token {token!r} already exists in the vocabulary.")
        if index < 0 or index > len(self.itos_):
            raise RuntimeError(
                f"Index {index} is out of range for vocabulary size {len(self.itos_)}."
            )

        self.itos_.insert(index, token)
        self._rebuild_stoi(start=index)

    def append_token(self, token: str) -> None:
        self.insert_token(token, len(self.itos_))

    def lookup_token(self, index: int) -> str:
        if index < 0 or index >= len(self.itos_):
            raise RuntimeError(
                f"Index {index} is out of range for vocabulary size {len(self.itos_)}."
            )
        try:
            return self.itos_[index]
        except IndexError as exc:
            raise RuntimeError(
                f"Index {index} is out of range for vocabulary size {len(self.itos_)}."
            ) from exc

    def lookup_tokens(self, indices: List[int]) -> List[str]:
        return [self.lookup_token(index) for index in indices]

    def lookup_indices(self, tokens: List[str]) -> List[int]:
        return [self[token] for token in tokens]

    def get_stoi(self) -> Dict[str, int]:
        return {token: index for index, token in enumerate(self.itos_)}

    def get_itos(self) -> List[str]:
        return list(self.itos_)

    def __prepare_scriptable__(self) -> "Vocab":
        return self

    def _rebuild_stoi(self, start: int = 0) -> None:
        self.stoi_ = {token: index for index, token in enumerate(self.itos_)}


def vocab(
    ordered_dict: Mapping[str, int],
    min_freq: int = 1,
    specials: Optional[List[str]] = None,
    special_first: bool = True,
) -> Vocab:
    """Build a Vocab from an ordered token-frequency mapping.

    This mirrors the subset of vocabulary factory behavior used by GeneVocab.
    """
    tokens: List[str] = []
    special_set = set(specials or [])

    if specials is not None and special_first:
        tokens.extend(specials)

    tokens.extend(
        token
        for token, freq in ordered_dict.items()
        if freq >= min_freq and token not in special_set
    )

    if specials is not None and not special_first:
        tokens.extend(specials)

    return Vocab(tokens)


def build_vocab_from_iterator(
    iterator: Iterable[Iterable[str]],
    min_freq: int = 1,
    specials: Optional[List[str]] = None,
    special_first: bool = True,
) -> Vocab:
    counter: Dict[str, int] = {}
    for tokens in iterator:
        for token in tokens:
            counter[token] = counter.get(token, 0) + 1

    sorted_items = sorted(counter.items(), key=lambda item: item[0])
    sorted_items.sort(key=lambda item: item[1], reverse=True)
    ordered_dict = OrderedDict(sorted_items)
    return vocab(ordered_dict, min_freq=min_freq, specials=specials, special_first=special_first)
