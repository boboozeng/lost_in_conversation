import json
import os
from pathlib import Path
import httpx
from openai import OpenAI, AzureOpenAI

class ModelRouter:
    def __init__(self, config_path=None):
        self.timeout_args = httpx.Timeout(connect=30.0, read=300.0, write=60.0, pool=10.0)
        self.clients = {}
        self.config_path = config_path or (Path(__file__).resolve().parent / "model_routing.json")
        self.config = self._load_config()
        self.aliases = self.config.get("aliases", {})


    def _load_config(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"Routing config not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def normalize_model_name(self, model):
        return self.aliases.get(model, model)

    def resolve_route(self, model):
        model = self.normalize_model_name(model)

        for route in self.config.get("routes", []):
            for prefix in route.get("match_prefix", []):
                if model.startswith(prefix):
                    return model, route

        return model, self.config["default"]

    def _client_cache_key(self, route):
        return (
            route.get("client_type", "openai"),
            route.get("api_key_env"),
            route.get("base_url"),
            route.get("azure_endpoint"),
            route.get("api_version"),
        )

    def _build_client(self, route):
        client_type = route.get("client_type", "openai")

        if client_type == "azure":
            api_key_env = route.get("api_key_env", "AZURE_OPENAI_API_KEY")
            azure_endpoint = route.get("azure_endpoint") or os.environ.get("AZURE_OPENAI_ENDPOINT")
            api_version = route.get("api_version", "2024-10-01-preview")

            assert api_key_env in os.environ, f"{api_key_env} environment variable must be set"
            assert azure_endpoint, "azure_endpoint must be configured for Azure route"

            return AzureOpenAI(
                api_key=os.environ[api_key_env],
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                timeout=self.timeout_args,
            )

        api_key_env = route.get("api_key_env", "OPENAI_API_KEY")
        assert api_key_env in os.environ, f"{api_key_env} environment variable must be set"

        kwargs = {
            "api_key": os.environ[api_key_env],
            "timeout": self.timeout_args,
        }
        if route.get("base_url"):
            kwargs["base_url"] = route["base_url"]

        return OpenAI(**kwargs)

    def get_client(self, model):
        normalized_model, route = self.resolve_route(model)
        cache_key = self._client_cache_key(route)

        if cache_key not in self.clients:
            self.clients[cache_key] = self._build_client(route)
            base_url = route.get("base_url") or "default OpenAI endpoint"
            print(f"[router] model={normalized_model} -> {base_url}")

        return normalized_model, self.clients[cache_key]
