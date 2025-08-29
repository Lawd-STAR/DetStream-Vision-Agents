#!/usr/bin/env python3
"""
Development CLI tool for stream-agents
Converted from Makefile - provides local testing capabilities extracted from GitHub workflows
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import click


def run_command(cmd: List[str], env: Optional[dict] = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command with proper error handling and output."""
    click.echo(f"Running: {' '.join(cmd)}")
    
    # Set up environment
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=False,
            env=full_env,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        if check:
            click.echo(f"Command failed with exit code {e.returncode}", err=True)
            sys.exit(e.returncode)
        return e


def discover_plugins() -> List[str]:
    """Discover available plugins by scanning the plugins directory."""
    plugins = ["core"]
    plugins_dir = Path("plugins")
    
    if plugins_dir.exists():
        for item in plugins_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                plugins.append(item.name)
    
    return plugins


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Development CLI tool for stream-agents local testing."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
def discover_plugins_cmd():
    """Discover available plugins and show them."""
    click.echo("Discovering plugins...")
    plugins = discover_plugins()
    
    click.echo("Available plugins:")
    for plugin in plugins:
        click.echo(f"  - {plugin}")


@cli.command()
def lint():
    """Run ruff linting."""
    click.echo("Running ruff lint...")
    run_command(["uv", "run", "ruff", "check", "."])


@cli.command()
@click.option("--plugin", default="core", help="Plugin to install extras for")
def install(plugin: str):
    """Install project dependencies with uv."""
    click.echo("Installing dependencies...")
    
    # Set environment variable
    env = {"PYTHONDONTWRITEBYTECODE": "1"}
    
    if Path("pyproject.toml").exists():
        click.echo("Installing with uv sync...")
        
        # Try full sync first, fallback to basic sync
        try:
            run_command(["uv", "sync", "--all-extras", "--dev"], env=env)
        except subprocess.CalledProcessError:
            click.echo("Full sync failed, trying basic sync...")
            run_command(["uv", "sync"], env=env)
        
        # Install plugin-specific extras if not core
        if plugin != "core":
            click.echo(f"Installing plugin-specific extras for {plugin}...")
            try:
                run_command(["uv", "sync", "--extra", plugin], env=env, check=False)
            except subprocess.CalledProcessError:
                click.echo(f"No extra found for plugin {plugin}, continuing...")
    else:
        click.echo("No pyproject.toml found, falling back to requirements.txt")
        if Path("requirements.txt").exists():
            run_command(["uv", "pip", "install", "-r", "requirements.txt"], env=env)
        else:
            click.echo("No requirements.txt found either")


@cli.command()
@click.option("--marker", default="not integration", help="Pytest marker expression")
@click.option("--plugin", default="core", help="Plugin to test")
@click.option("--extra-args", default="", help="Extra pytest arguments")
@click.option("--install-deps/--no-install", default=True, help="Install dependencies before testing")
def test(marker: str, plugin: str, extra_args: str, install_deps: bool):
    """Run tests with specified marker."""
    if install_deps:
        ctx = click.get_current_context()
        ctx.invoke(install, plugin=plugin)
    
    click.echo(f"Running tests with marker: {marker}")
    click.echo(f"Plugin: {plugin}")
    click.echo(f"Extra args: {extra_args}")
    
    # Set up environment
    env = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PLUGIN": plugin
    }
    
    # Build pytest command
    cmd = ["uv", "run", "pytest", "-m", marker]
    if extra_args:
        cmd.extend(extra_args.split())
    
    run_command(cmd, env=env)


@cli.command()
@click.option("--extra-args", default="", help="Extra pytest arguments")
@click.option("--install-deps/--no-install", default=True, help="Install dependencies before testing")
def test_unit(extra_args: str, install_deps: bool):
    """Run unit tests only."""
    ctx = click.get_current_context()
    ctx.invoke(test, marker="not integration", plugin="core", extra_args=extra_args, install_deps=install_deps)


@cli.command()
@click.option("--extra-args", default="", help="Extra pytest arguments")
@click.option("--install-deps/--no-install", default=True, help="Install dependencies before testing")
def test_integration(extra_args: str, install_deps: bool):
    """Run integration tests only."""
    ctx = click.get_current_context()
    ctx.invoke(test, marker="integration", plugin="core", extra_args=extra_args, install_deps=install_deps)


@cli.command()
@click.option("--plugin", default="core", help="Plugin to test")
@click.option("--extra-args", default="", help="Extra pytest arguments")
@click.option("--install-deps/--no-install", default=True, help="Install dependencies before testing")
def test_plugin(plugin: str, extra_args: str, install_deps: bool):
    """Run tests for a specific plugin."""
    if plugin == "core":
        click.echo("Running core tests...")
    else:
        click.echo(f"Running tests for plugin: {plugin}")
    
    ctx = click.get_current_context()
    ctx.invoke(test, marker="not integration", plugin=plugin, extra_args=extra_args, install_deps=install_deps)


@cli.command()
@click.option("--install-deps/--no-install", default=True, help="Install dependencies before testing")
def test_all(install_deps: bool):
    """Run all tests (unit + integration)."""
    click.echo("Running all tests...")
    ctx = click.get_current_context()
    
    ctx.invoke(test_unit, extra_args="", install_deps=install_deps)
    ctx.invoke(test_integration, extra_args="", install_deps=False)  # Don't reinstall


@cli.command()
@click.option("--install-deps/--no-install", default=True, help="Install dependencies before testing")
def test_all_plugins(install_deps: bool):
    """Run tests for all available plugins."""
    click.echo("Testing all plugins...")
    plugins = discover_plugins()
    ctx = click.get_current_context()
    
    first_plugin = True
    for plugin in plugins:
        click.echo(f"Testing plugin: {plugin}")
        try:
            ctx.invoke(test_plugin, plugin=plugin, extra_args="", install_deps=install_deps and first_plugin)
            first_plugin = False  # Only install deps for first plugin
        except SystemExit:
            click.echo(f"Tests failed for plugin {plugin}, continuing with next plugin...")
            continue


@cli.command()
def setup():
    """Initial setup - install dependencies and run basic checks."""
    click.echo("Setting up development environment...")
    ctx = click.get_current_context()
    
    ctx.invoke(install)
    ctx.invoke(discover_plugins_cmd)
    ctx.invoke(lint)
    
    click.echo("Setup complete! Try 'python dev.py test-unit' to run unit tests.")


@cli.command()
def clean():
    """Clean up temporary files."""
    click.echo("Cleaning up...")
    
    # Find and remove Python cache files
    for pattern in ["**/*.pyc", "**/__pycache__", "**/*.egg-info", "**/.coverage", "**/.pytest_cache"]:
        for path in Path(".").glob(pattern):
            if path.is_file():
                path.unlink()
                click.echo(f"Removed file: {path}")
            elif path.is_dir():
                import shutil
                shutil.rmtree(path)
                click.echo(f"Removed directory: {path}")


@cli.command()
@click.option("--extra-args", default="-x --tb=short", help="Extra pytest arguments")
def dev_test(extra_args: str):
    """Quick development test (unit tests only)."""
    ctx = click.get_current_context()
    ctx.invoke(test_unit, extra_args=extra_args, install_deps=True)


@cli.command()
def dev_lint():
    """Quick lint check."""
    ctx = click.get_current_context()
    ctx.invoke(lint)


@cli.command()
def dev_check():
    """Quick development check (lint + unit tests)."""
    ctx = click.get_current_context()
    ctx.invoke(dev_lint)
    ctx.invoke(dev_test)


@cli.command()
def example_xai():
    """Example: Run tests for xai plugin."""
    ctx = click.get_current_context()
    ctx.invoke(test_plugin, plugin="xai")


@cli.command()
def example_integration():
    """Example: Run integration tests with verbose output."""
    ctx = click.get_current_context()
    ctx.invoke(test_integration, extra_args="-v -s")


@cli.command()
def example_specific():
    """Example: Show how to run specific test file."""
    click.echo("Example: uv run pytest tests/test_specific.py -v")
    click.echo("You can run: python dev.py test --extra-args 'tests/test_specific.py -v'")


@cli.command()
def ci_unit():
    """Run unit tests like CI."""
    ctx = click.get_current_context()
    ctx.invoke(install)
    ctx.invoke(lint)
    ctx.invoke(test_unit, install_deps=False)


@cli.command()
def ci_integration():
    """Run integration tests like CI."""
    ctx = click.get_current_context()
    ctx.invoke(install)
    ctx.invoke(test_integration, install_deps=False)


@cli.command()
def ci_full():
    """Run full CI pipeline locally."""
    ctx = click.get_current_context()
    ctx.invoke(install)
    ctx.invoke(lint)
    ctx.invoke(test_all, install_deps=False)
    ctx.invoke(test_all_plugins, install_deps=False)


if __name__ == "__main__":
    cli()
