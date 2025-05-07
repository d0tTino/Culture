"""
Discord bot interface for the Culture simulation.
Provides real-time updates about the simulation to a Discord channel.
"""

import asyncio
import logging
import discord
from typing import Optional, Union

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
                embed = discord.Embed(
                    title="🤖 Culture Simulation Bot Online",
                    description="Connected and ready to provide simulation updates!",
                    color=discord.Color.blue()
                )
                embed.set_footer(text=f"Channel ID: {self.channel_id}")
                await channel.send(embed=embed)
            else:
                logger.warning(f"Could not find Discord channel with ID: {self.channel_id}")
    
    async def send_simulation_update(self, content: Optional[str] = None, embed: Optional[discord.Embed] = None):
        """
        Send a simulation update message to the configured Discord channel.
        
        Args:
            content (Optional[str]): The text message content to send
            embed (Optional[discord.Embed]): The embed object to send
            
        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        if not self.is_ready:
            logger.warning("Discord bot not ready yet, message not sent")
            return False
        
        try:
            channel = self.client.get_channel(self.channel_id)
            if not channel:
                logger.warning(f"Could not find Discord channel with ID: {self.channel_id}")
                return False
                
            if embed:
                await channel.send(embed=embed)
                logger.debug("Sent Discord embed update")
                return True
            elif content:
                # Discord message length limit is 2000 characters
                if len(content) > 1990:  # Leave some room for prefixes/suffixes
                    content = content[:1990] + "..."
                await channel.send(content)
                logger.debug(f"Sent Discord text update: {content[:50]}...")
                return True
            else:
                logger.warning("send_simulation_update called with no content or embed")
                return False
        except Exception as e:
            logger.error(f"Error sending Discord message: {e}")
            return False
    
    def create_step_start_embed(self, step: int) -> discord.Embed:
        """Creates an embed for simulation step start"""
        embed = discord.Embed(
            title=f"📊 Simulation Step {step} Started",
            color=discord.Color.blue()
        )
        return embed
    
    def create_step_end_embed(self, step: int, agent_summaries: list) -> discord.Embed:
        """Creates an embed for simulation step end with agent summaries"""
        embed = discord.Embed(
            title=f"✅ Simulation Step {step} Completed",
            description="Agent Status Summary:",
            color=discord.Color.green()
        )
        
        # Add agent summaries as fields
        for summary in agent_summaries:
            embed.add_field(name=f"Agent Status", value=summary, inline=False)
            
        return embed
    
    def create_knowledge_board_embed(self, agent_id: str, content: str, step: int) -> discord.Embed:
        """Creates an embed for Knowledge Board posts"""
        embed = discord.Embed(
            title=f"📝 New Knowledge Board Entry (Step {step})",
            description=f"```{content}```",
            color=discord.Color.gold()
        )
        embed.set_author(name=f"Posted by Agent {agent_id[:8]}")
        return embed
    
    def create_role_change_embed(self, agent_id: str, old_role: str, new_role: str, step: int) -> discord.Embed:
        """Creates an embed for agent role changes"""
        embed = discord.Embed(
            title=f"🔄 Agent Role Change (Step {step})",
            description=f"Agent {agent_id[:8]} changed from **{old_role}** to **{new_role}**",
            color=discord.Color.purple()
        )
        return embed
    
    def create_project_embed(self, action: str, project_name: str, project_id: str, agent_id: str, step: int) -> discord.Embed:
        """Creates an embed for project creation/joining/leaving"""
        if action == "create":
            title = f"🏗️ New Project Created (Step {step})"
            description = f"Agent {agent_id[:8]} created project **{project_name}** (ID: {project_id})"
            color = discord.Color.teal()
        elif action == "join":
            title = f"➕ Agent Joined Project (Step {step})"
            description = f"Agent {agent_id[:8]} joined project **{project_name}** (ID: {project_id})"
            color = discord.Color.dark_green()
        elif action == "leave":
            title = f"➖ Agent Left Project (Step {step})"
            description = f"Agent {agent_id[:8]} left project **{project_name}** (ID: {project_id})"
            color = discord.Color.dark_orange()
        else:
            title = f"🏢 Project Update (Step {step})"
            description = f"Project **{project_name}** (ID: {project_id}) was updated"
            color = discord.Color.light_grey()
            
        embed = discord.Embed(
            title=title,
            description=description,
            color=color
        )
        return embed
    
    def create_agent_message_embed(self, sender_id: str, content: str, step: int, recipient_id: Optional[str] = None) -> discord.Embed:
        """Creates an embed for agent messages (broadcast or targeted)"""
        target_info = f"to Agent {recipient_id[:8]}" if recipient_id else "to All (Broadcast)"
        embed = discord.Embed(
            title=f"💬 Agent Message (Step {step})",
            description=f"```{content}```",
            color=discord.Color.blue()
        )
        embed.set_author(name=f"From Agent {sender_id[:8]} {target_info}")
        return embed
    
    def create_ip_change_embed(self, agent_id: str, old_ip: int, new_ip: int, reason: str, step: int) -> discord.Embed:
        """Creates an embed for influence point changes"""
        change = new_ip - old_ip
        change_text = f"+{change}" if change > 0 else f"{change}"
        color = discord.Color.green() if change > 0 else discord.Color.red()
        
        embed = discord.Embed(
            title=f"💰 Influence Points Change (Step {step})",
            description=f"Agent {agent_id[:8]} IP: {old_ip} → {new_ip} ({change_text})",
            color=color
        )
        embed.add_field(name="Reason", value=reason, inline=False)
        return embed
    
    def create_du_change_embed(self, agent_id: str, old_du: float, new_du: float, reason: str, step: int) -> discord.Embed:
        """Creates an embed for decision unit changes"""
        change = new_du - old_du
        change_text = f"+{change:.2f}" if change > 0 else f"{change:.2f}"
        color = discord.Color.green() if change > 0 else discord.Color.red()
        
        embed = discord.Embed(
            title=f"💾 Decision Units Change (Step {step})",
            description=f"Agent {agent_id[:8]} DU: {old_du:.2f} → {new_du:.2f} ({change_text})",
            color=color
        )
        embed.add_field(name="Reason", value=reason, inline=False)
        return embed
    
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
            if self.is_ready:
                channel = self.client.get_channel(self.channel_id)
                if channel:
                    embed = discord.Embed(
                        title="🛑 Simulation Complete",
                        description="The Culture simulation has ended. Bot going offline.",
                        color=discord.Color.red()
                    )
                    await channel.send(embed=embed)
            await self.client.close()
            self.is_ready = False
            logger.info("Discord bot stopped")
        except Exception as e:
            logger.error(f"Error stopping Discord bot: {e}") 