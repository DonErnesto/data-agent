@dataclass(frozen=True)
class Goal:
    priority: int
    name: str
    description: str