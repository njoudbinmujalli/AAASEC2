import sys
import httpx


def discover(peer_base_url: str) -> dict:
    """Fetch a peer agent's Agent Card and print its identity + skills."""
    url = f"{peer_base_url.rstrip('/')}/.well-known/agent-card.json"
    response = httpx.get(url, trust_env=False)
    response.raise_for_status()
    card = response.json()

    print(f"Discovered: {card['name']}")
    print(f"Description: {card['description']}")
    print("Skills:")
    for skill in card.get("skills", []):
        print(f"  - {skill['id']}: {skill['description']}")

    return card


def delegate(card: dict, task: str) -> str:
    """Send a task to the peer agent, using the URL from ITS card."""
    endpoint = card["url"]
    response = httpx.post(
        endpoint,
        json={"input": task},
        timeout=120.0,
        trust_env=False,
    )

    response.raise_for_status()
    result = response.json()

    output_text = result["output"][0]["content"][0]["text"]
    return output_text


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: uv run python src/a2a_client.py <peer_url> \"task\"")
        sys.exit(1)

    peer_url = sys.argv[1]
    task = sys.argv[2]

    card = discover(peer_url)
    result = delegate(card, task)

    print("\n--- Response ---")
    print(result)