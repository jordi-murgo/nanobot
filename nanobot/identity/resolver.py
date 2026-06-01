"""Pure path helper functions for user and team workspaces."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from nanobot.config.schema import GroupIdentity, LogicalUser, TeamConfig, TeamsConfig, UsersConfig


def derive_user_workspace_path(base: Path, user_name: str) -> Path:
    """Return path to user-scoped workspace. Does NOT create the directory.

    Args:
        base: Base workspace directory.
        user_name: Logical user name (e.g. 'jordi').

    Returns:
        Path to user workspace (base / "users" / user_name).
    """
    return base / "users" / user_name


def derive_team_workspace_path(base: Path, team_slug: str) -> Path:
    """Return path to team-scoped workspace. Does NOT create the directory.

    Args:
        base: Base workspace directory.
        team_slug: Team slug (e.g. 'familia').

    Returns:
        Path to team workspace (base / "teams" / team_slug).
    """
    return base / "teams" / team_slug


def find_logical_user(
    users_config: UsersConfig, channel: str, sender_id: str
) -> LogicalUser | None:
    """Find a logical user by channel+sender_id identity.

    Args:
        users_config: Users configuration.
        channel: Channel name (e.g. 'telegram', 'whatsapp').
        sender_id: Channel-specific sender ID.

    Returns:
        LogicalUser if found, None otherwise.
    """
    base_id = sender_id.split("|")[0] if "|" in sender_id else sender_id
    for user in users_config.users:
        for identity in user.identities:
            if identity.channel == channel and identity.id in (sender_id, base_id):
                return user
    return None


def find_team_by_group(
    teams_config: TeamsConfig, group_id: str, channel: str | None = None
) -> TeamConfig | None:
    for team in teams_config.teams:
        for group in team.groups:
            if isinstance(group, GroupIdentity):
                if group.id != group_id:
                    continue
                if channel is None or group.channel == channel or group.channel == "*":
                    return team
            elif group == group_id:
                return team
    return None


def resolve_team(
    teams_config: TeamsConfig,
    logical_user: LogicalUser | None,
    group_id: str | None,
    session_pinned_team: str | None = None,
    channel: str | None = None,
) -> str | None:
    """Resolve active team slug using deterministic precedence chain.

    Precedence (highest to lowest):
    1. Group mapping: if group_id is in teams[*].groups → return that team slug
    2. Session-pinned team: if session_pinned_team is set AND user is still a member
    3. Single-team shortcut: if user belongs to exactly one team and no group context
    4. Ambiguous (multi-team, no group): log warning, return None

    Returns None gracefully on ambiguity — never raises, never guesses.
    """
    if group_id is not None:
        mapped_team = find_team_by_group(teams_config, group_id, channel=channel)
        if mapped_team is not None:
            return mapped_team.slug

    if session_pinned_team is not None and logical_user is not None:
        eligible_team_slugs = {
            team.slug for team in teams_config.teams if logical_user.name in team.members
        }
        if session_pinned_team in eligible_team_slugs:
            return session_pinned_team

    if logical_user is None:
        return None

    user_teams = [team for team in teams_config.teams if logical_user.name in team.members]
    if len(user_teams) == 1:
        return user_teams[0].slug

    if len(user_teams) > 1:
        logger.warning(
            "Ambiguous team for user {}: belongs to {} teams, no group context",
            logical_user.name,
            len(user_teams),
        )
        return None

    return None


def derive_team_artifact_path(base: Path, team_slug: str, artifact_type: str) -> Path:
    """Return path to team artifact directory. Does NOT create the directory.

    Args:
        base: Base workspace directory.
        team_slug: Team slug (e.g. 'familia').
        artifact_type: Artifact type (e.g. 'shopping', 'calendar').

    Returns:
        Path to team artifact (base / "teams" / team_slug / "artifacts" / artifact_type).
    """
    return base / "teams" / team_slug / "artifacts" / artifact_type


def get_effective_workspace_paths(
    workspace: Path,
    system_workspace: Path,
    user_name: str | None = None,
    team_slug: str | None = None,
) -> dict[str, Path]:
    """Return a dict with all effective workspace paths.

    Args:
        workspace: Base runtime workspace directory.
        system_workspace: System workspace directory (e.g. /ws/system).
        user_name: Optional logical user name (e.g. 'jordi').
        team_slug: Optional team slug (e.g. 'familia').

    Returns:
        Dict with keys: "system", "user", "team", "runtime"
        - system: system_workspace path
        - user: workspace / "users" / user_name (only if user_name provided)
        - team: workspace / "teams" / team_slug (only if team_slug provided)
        - runtime: workspace
    """
    result: dict[str, Path] = {
        "system": system_workspace,
        "runtime": workspace,
    }

    if user_name is not None:
        result["user"] = derive_user_workspace_path(workspace, user_name)

    if team_slug is not None:
        result["team"] = derive_team_workspace_path(workspace, team_slug)

    return result
