"""
Discord bot interface for the Culture simulation.
Provides real-time updates about the simulation to a Discord channel.
"""

import asyncio
import logging
import discord
from typing import Optional

logger = logging.getLogger(__name__)

class SimulationDiscordBot:
    """
    A Discord bot that provides real-time updates about the Culture simulation.
    
    This bot connects to a specified Discord channel and sends read-only updates
    about simulation events, including Knowledge Board updates, agent messages,
    role changes, and other significant state changes.
    """
    
    def __init__(self, bot_token: str, channel_id: int):
        """
        Initialize the Discord bot with token and target channel.
        
        Args:
            bot_token (str): The Discord bot token for authentication
            channel_id (int): The ID of the Discord channel to send updates to
        """
        self.bot_token = bot_token
        self.channel_id = channel_id
        self.is_ready = False
        
        # Set up intents (permissions)
        intents = discord.Intents.default()
        intents.message_content = True  # Enable if you plan to add commands later
        
        # Create Discord client
        self.client = discord.Client(intents=intents)
        
        # Set up event handlers
        @self.client.event
        async def on_ready():
            """Event handler that fires when the bot connects to Discord."""
            self.is_ready = True
            logger.info(f"Discord bot {self.client.user} connected and ready!")
            
            # Get the target channel and send a startup message
            channel = self.client.get_channel(self.channel_id)
            if channel:
                await channel.send("🤖 Culture Simulation Bot connected! Starting to monitor simulation...")
            else:
                logger.warning(f"Could not find Discord channel with ID: {self.channel_id}")
    
    async def send_simulation_update(self, message: str):
        """
        Send a simulation update message to the configured Discord channel.
        
        Args:
            message (str): The message content to send
            
        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        if not self.is_ready:
            logger.warning("Discord bot not ready yet, message not sent")
            return False
        
        try:
            channel = self.client.get_channel(self.channel_id)
            if channel:
                await channel.send(message)
                logger.debug(f"Sent Discord update: {message[:50]}...")
                return True
            else:
                logger.warning(f"Could not find Discord channel with ID: {self.channel_id}")
                return False
        except Exception as e:
            logger.error(f"Error sending Discord message: {e}")
            return False
    
    async def run_bot(self):
        """
        Start the Discord bot and connect to Discord.
        This is a blocking call that should be run in an asyncio task.
        """
        try:
            logger.info(f"Starting Discord bot, connecting to channel ID: {self.channel_id}")
            await self.client.start(self.bot_token)
        except Exception as e:
            logger.error(f"Error starting Discord bot: {e}")
    
    async def stop_bot(self):
        """Stop the Discord bot and close the connection."""
        try:
            logger.info("Stopping Discord bot...")
            await self.client.close()
            self.is_ready = False
            logger.info("Discord bot stopped")
        except Exception as e:
            logger.error(f"Error stopping Discord bot: {e}") 