"""mbx-runner — bring up a bound myllmbox box: vLLM container + keepalive auth proxy + cloudflared."""
from __future__ import annotations

import json
import logging
import subprocess

import click

from . import config, supervisor


@click.group()
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")


@main.command()
@click.option("--config", "yaml_path", default="myllmbox.yaml", show_default=True, help="box config YAML")
@click.option("--proxy-port", type=int, default=None, help="override proxy.port")
@click.option("--recipe", "recipe_file", default=None, help="bring the model up via spark-vllm-docker run-recipe (recipe name/path)")
@click.option("--attach", is_flag=True, help="manage nothing model-side — proxy+tunnel onto an already-serving upstream")
@click.option("--upstream-port", type=int, default=None, help="where the model listens (attach/recipe)")
def up(yaml_path: str, proxy_port: int | None, recipe_file: str | None, attach: bool, upstream_port: int | None) -> None:
    """Start the box: model (built-in vLLM · recipe · attach) + keepalive proxy + tunnel."""
    cli: dict = {}
    if proxy_port:
        cli["proxy"] = {"port": proxy_port}
    if recipe_file:
        cli["mode"] = "recipe"
        cli["recipe"] = {"file": recipe_file}
    if attach:
        cli["mode"] = "attach"
    if upstream_port:
        cli["upstream_port"] = upstream_port
    supervisor.up(config.load(cli=cli, yaml_path=yaml_path))


@main.command()
def down() -> None:
    """Stop tunnel + proxy + vLLM (idempotent)."""
    supervisor.down()


@main.command()
@click.option("--config", "yaml_path", default="myllmbox.yaml", show_default=True)
def status(yaml_path: str) -> None:
    """Report each process + the model card, locally and (with PUBLIC_URL) through the edge."""
    click.echo(json.dumps(supervisor.status(config.load(yaml_path=yaml_path)), indent=2))


@main.group()
def recipes() -> None:
    """Manage recipe packs — folders of other people's model-serving scripts, in one designated place."""


@recipes.command("add")
@click.argument("git_url")
@click.argument("name", required=False)
@click.option("--config", "yaml_path", default="myllmbox.yaml", show_default=True)
def recipes_add(git_url: str, name: str | None, yaml_path: str) -> None:
    """Clone a recipe pack (someone's script collection) into the recipes root."""
    import pathlib

    root = pathlib.Path(_recipes_root(yaml_path)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    dest = root / (name or git_url.rstrip("/").removesuffix(".git").rsplit("/", 1)[-1])
    subprocess.run(["git", "clone", git_url, str(dest)], check=True)
    click.echo(f"pack installed: {dest.name} → use recipes with --recipe {dest.name}/<recipe>")


@recipes.command("list")
@click.option("--config", "yaml_path", default="myllmbox.yaml", show_default=True)
def recipes_list(yaml_path: str) -> None:
    """List installed packs and their recipes."""
    import pathlib

    root = pathlib.Path(_recipes_root(yaml_path)).expanduser()
    if not root.exists():
        click.echo(f"(no packs — recipes root {root} is empty; add one: mbx-runner recipes add <git-url>)")
        return
    for pack in sorted(p for p in root.iterdir() if p.is_dir()):
        runnable = "✓" if (pack / "run-recipe.sh").exists() else "✗ (no run-recipe.sh)"
        click.echo(f"{pack.name} {runnable}")
        for y in sorted(pack.rglob("*.yaml"))[:20]:
            click.echo(f"  {pack.name}/{y.relative_to(pack)}")


def _recipes_root(yaml_path: str) -> str:
    import os
    cfg = config.load(env={**os.environ, "TUNNEL_TOKEN": os.environ.get("TUNNEL_TOKEN", "-")}, yaml_path=yaml_path)
    return cfg["recipes"]["root"]


@main.command()
@click.option("--follow", "-f", is_flag=True)
def logs(follow: bool) -> None:
    """Tail the proxy + cloudflared logs (vLLM's live in `docker logs mbx-vllm`)."""
    args = ["tail", "-n", "50"] + (["-f"] if follow else []) + [".mbx/proxy.log", ".mbx/cloudflared.log"]
    subprocess.run(args, check=False)


if __name__ == "__main__":
    main()
