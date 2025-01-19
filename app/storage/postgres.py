from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, update
import logging

from ..models.schemas import RawListing, ProcessedListing

logger = logging.getLogger(__name__)

class PostgresClient:
    def __init__(self, connection_url: str):
        self.engine = create_async_engine(connection_url)
        self.async_session = sessionmaker(
            self.engine, 
            class_=AsyncSession, 
            expire_on_commit=False
        )

    async def initialize(self):
        """Initialize database connection"""
        # Create tables if they don't exist
        async with self.engine.begin() as conn:
            # await conn.run_sync(Base.metadata.create_all)
            pass

    async def cleanup(self):
        """Cleanup database connections"""
        await self.engine.dispose()

    async def fetch_unprocessed_listings(self, limit: int) -> List[RawListing]:
        """Fetch unprocessed listings from the database"""
        async with self.async_session() as session:
            result = await session.execute(
                select(RawListing)
                .where(RawListing.processed == False)
                .limit(limit)
            )
            return result.scalars().all()

    async def store_processed_listings(self, listings: List[ProcessedListing]):
        """Store processed listings in the database"""
        async with self.async_session() as session:
            session.add_all(listings)
            await session.commit()

    async def mark_listings_processed(self, listing_ids: List[int]):
        """Mark raw listings as processed"""
        async with self.async_session() as session:
            await session.execute(
                update(RawListing)
                .where(RawListing.id.in_(listing_ids))
                .values(processed=True)
            )
            await session.commit()

    async def transaction(self):
        """Context manager for transactions"""
        return self.async_session() 