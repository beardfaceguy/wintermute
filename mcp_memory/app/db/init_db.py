from app.models.memory_entry import Base
from app.db.session import engine


def init_db():
    Base.metadata.create_all(bind=engine)
