# typings/pgvector/sqlalchemy.py


class Vector:
    dimensions: int

    def __init__(self, dimensions: int): ...
