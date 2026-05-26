"""Hand-curated entity dataset for M4 feature derivation.

Ferrando v2 (arXiv:2411.14257) §3 uses ~8k entities per type (players, movies,
cities, songs) from Wikidata, classified known/unknown by querying the model's
attribute recall accuracy. For v2 we use a much smaller hand-curated set: ~30
entities per type, each pre-labeled as known (well-known to a 9B base model) or
unknown (plausible but non-existent / extremely obscure).

Why hand-curated rather than the full Wikidata pipeline:
	1. Ferrando's pipeline assumes a model_alias system and is ~half-day to
	   re-run on Gemma-2-9B at our target layer (L21).
	2. The unknown-entity latent is robust — Ferrando shows it generalizes
	   across entity types; we don't need 8k entities per type to identify it.
	3. v2 is a workshop paper, not a benchmark.
	4. Reproducible: the dataset is committed in code, no Wikidata download.

If the derived latent's separation score is weak (< 0.4 min-across-types) on
this small set, fall back to running Ferrando's pipeline at L21 — that's a
budget call (~$15 extra A100).

Reference: Ferrando v2 §3, Example 5 prompt template, Appendix A classification
criterion (≥2 attributes correct = known, all wrong = unknown).
"""

from dataclasses import dataclass
from typing import Iterable


# entities chosen for Gemma-2-9B-PT's pretraining cutoff (Aug 2024). Known entities
# are unambiguously famous; unknown entities are either plausible-sounding but
# non-existent, mildly-misspelled famous ones (Ferrando's table 1 trick), or extreme
# obscurities. The known/unknown split is by author judgement — the feature
# derivation script will report per-entity SAE latent activations so you can spot
# any borderline misclassifications post-hoc.
KNOWN_PLAYERS = [
	"Michael Jordan", "LeBron James", "Kobe Bryant", "Magic Johnson",
	"Larry Bird", "Shaquille O'Neal", "Kareem Abdul-Jabbar", "Wilt Chamberlain",
	"Bill Russell", "Tim Duncan", "Hakeem Olajuwon", "Karl Malone",
	"Charles Barkley", "John Stockton", "Allen Iverson", "Kevin Durant",
	"Stephen Curry", "Dirk Nowitzki", "Dwyane Wade", "Chris Paul",
	"James Harden", "Russell Westbrook", "Kawhi Leonard", "Damian Lillard",
	"Anthony Davis", "Giannis Antetokounmpo", "Nikola Jokic", "Luka Doncic",
	"Joel Embiid", "Jayson Tatum",
]
UNKNOWN_PLAYERS = [
	# plausible names that aren't real NBA players (Ferrando's trick: phonotactic plausibility)
	"Wilson Brown", "Marcus Henson", "Tyler Whitfield", "Derek Caldwell",
	"Jonathan Pierce", "Anthony Marsh", "Brandon Reeves", "Kevin Spencer",
	"Christopher Vaughn", "Robert Knowles", "Daniel Larkin", "Steven Ashcroft",
	"Patrick Ellsworth", "Michael Drayton", "James Holloway", "David Rutland",
	"Andrew Pemberton", "Matthew Kingsford", "Joseph Wexler", "Thomas Yardley",
	# mildly misspelled famous ones — should activate "unknown" if model is strict
	"Michael Joordan", "LeBon James", "Koby Bryant", "Magic Johnsone",
	"Shakwille O'Neil", "Kareem Abdul-Jabbur", "Wilt Chamberlin", "Bil Russel",
	"Tim Dunkan", "Hakem Olajuwan",
]


KNOWN_MOVIES = [
	"12 Angry Men", "The Godfather", "Citizen Kane", "Casablanca",
	"Pulp Fiction", "The Shawshank Redemption", "Schindler's List", "Forrest Gump",
	"The Dark Knight", "Inception", "Goodfellas", "The Matrix",
	"Fight Club", "Star Wars", "Jaws", "Titanic",
	"Apocalypse Now", "Taxi Driver", "Vertigo", "Psycho",
	"2001: A Space Odyssey", "Lawrence of Arabia", "Singin' in the Rain", "Gone with the Wind",
	"It's a Wonderful Life", "The Wizard of Oz", "Some Like It Hot", "On the Waterfront",
	"Sunset Boulevard", "Rear Window",
]
UNKNOWN_MOVIES = [
	"20 Angry Men", "The Godfaller", "Citizen Krane", "Casablonca",
	"Pulped Fiction", "The Shawshunk Redemption", "Schinder's List", "Forest Gumb",
	"The Dim Knight", "Incepture", "Goodfollas", "The Matrice",
	# plausible non-existent titles
	"The Quiet Witness", "Bridges of Algorin", "Last Train to Penmoor", "The Cobalt Hour",
	"Northern Reverie", "Ashes of Velmont", "The Velvet Conductor", "Glass Above the Sea",
	"Marigold Station", "The Cassidy Letters", "Departure at Eight", "The Aurelian Bond",
	"Halcyon Days of Mr. Drey", "The Burning Garden", "Letters from Tarmaston", "The Bellwether Inn",
	"Echoes of Innisfeld", "The Sandhammer Estate",
]


KNOWN_CITIES = [
	"Paris", "London", "New York", "Tokyo",
	"Beijing", "Moscow", "Cairo", "Sydney",
	"Mumbai", "Sao Paulo", "Mexico City", "Buenos Aires",
	"Istanbul", "Berlin", "Rome", "Madrid",
	"Amsterdam", "Stockholm", "Vienna", "Athens",
	"Bangkok", "Singapore", "Seoul", "Jakarta",
	"Toronto", "Vancouver", "Chicago", "Los Angeles",
	"San Francisco", "Boston",
]
UNKNOWN_CITIES = [
	# misspellings
	"Pareis", "Londun", "Tokyou", "Cairoh",
	"Mumby", "Berlyn", "Roam", "Madryd",
	# plausible non-existent
	"Anthon", "Velmoor", "Drayhampton", "Kelsmere",
	"Northpoint Ridge", "South Halverton", "New Tashbridge", "Old Penmoor",
	"Carrowfield", "Glassmere", "Sandhammer", "Yarrowstead",
	"Holloway Crossing", "Penmoor Heights", "Ashridge Bend", "Cobalt Springs",
	"Marigold Falls", "Tarmaston", "Innisfeld", "Velmont",
	"Aurelian Hills", "Bellwether",
]


KNOWN_SONGS = [
	"Yellow Submarine", "Imagine", "Bohemian Rhapsody", "Hey Jude",
	"Stairway to Heaven", "Hotel California", "Smells Like Teen Spirit", "Like a Rolling Stone",
	"Born to Run", "Purple Haze", "Respect", "What's Going On",
	"Good Vibrations", "Johnny B. Goode", "I Want to Hold Your Hand", "Satisfaction",
	"My Generation", "Yesterday", "Let It Be", "A Day in the Life",
	"Heartbreak Hotel", "Blue Suede Shoes", "Be My Baby", "California Dreamin'",
	"Born in the U.S.A.", "Thriller", "Billie Jean", "Smells Like Teen Spirit",
	"Wonderwall", "Karma Police",
]
UNKNOWN_SONGS = [
	"Turquoise Submarine", "Imagined", "Bohemian Rhapsidy", "Hey Joed",
	"Stairway to Heavin", "Hotell California", "Smells Like Teem Spirit", "Like a Rollin Stone",
	# plausible non-existent
	"The Velvet Drift", "Marigold Avenue", "Aurelian Skies", "Penmoor Lullaby",
	"Glass Above the River", "Sandhammer Sunday", "The Cobalt Hour", "Letters to Innisfeld",
	"Northern Reverie", "Ashes of Velmont", "The Quiet Witness Theme", "Bridges of Algorin",
	"Cassidy's Refrain", "Departure at Eight", "Echoes of Tarmaston", "Halcyon Days",
	"The Burning Garden", "Marigold Station", "The Bellwether Waltz", "Yarrowstead Blues",
	"Old Penmoor Reel", "Carrowfield Rag",
]


ENTITY_TYPES = ("player", "movie", "city", "song")


# Ferrando's prompt template (Example 5 in §3):
#   "The {entity_type} {entity_name} {relation}"
# where {relation} is a recall cue like "was born in" / "was directed by".
# We use ONE relation per entity type — the entity's last token is what we hook,
# so the relation only matters for the model's downstream attempt; the residual at
# the entity-last-token captures whatever the model "thinks" before recall starts.
_TEMPLATES = {
	"player": "The basketball player {name} was born in",
	"movie": "The movie {name} was directed by",
	"city": "The city {name} is located in the country of",
	"song": "The song {name} was originally performed by",
}


@dataclass
class EntityPrompt:
	entity_type: str        # one of ENTITY_TYPES
	entity_name: str        # the literal name to be substituted
	is_known: bool          # author-labeled
	prompt: str             # full prompt with template applied


def _entities(entity_type: str) -> tuple[list[str], list[str]]:
	"""Return (known_names, unknown_names) for one entity type."""
	mapping = {
		"player": (KNOWN_PLAYERS, UNKNOWN_PLAYERS),
		"movie": (KNOWN_MOVIES, UNKNOWN_MOVIES),
		"city": (KNOWN_CITIES, UNKNOWN_CITIES),
		"song": (KNOWN_SONGS, UNKNOWN_SONGS),
	}
	if entity_type not in mapping:
		raise ValueError(f"unknown entity_type {entity_type!r}; expected one of {ENTITY_TYPES}")
	return mapping[entity_type]


def build_prompts(entity_types: Iterable[str] = ENTITY_TYPES) -> list[EntityPrompt]:
	"""Build the full prompt set for the requested entity types."""
	prompts: list[EntityPrompt] = []
	for et in entity_types:
		known, unknown = _entities(et)
		template = _TEMPLATES[et]
		for name in known:
			prompts.append(EntityPrompt(et, name, True, template.format(name=name)))
		for name in unknown:
			prompts.append(EntityPrompt(et, name, False, template.format(name=name)))
	return prompts


def counts_per_type() -> dict[str, dict[str, int]]:
	"""Sanity check: how many known/unknown per type."""
	out: dict[str, dict[str, int]] = {}
	for et in ENTITY_TYPES:
		known, unknown = _entities(et)
		out[et] = {"known": len(known), "unknown": len(unknown)}
	return out


if __name__ == "__main__":
	print("entity counts per type:")
	for et, c in counts_per_type().items():
		print(f"  {et:8s}  known={c['known']}  unknown={c['unknown']}  total={c['known'] + c['unknown']}")

	print("\nfirst 3 prompts per type:")
	for et in ENTITY_TYPES:
		print(f"\n  {et.upper()}:")
		for p in build_prompts([et])[:3]:
			tag = "KNOWN" if p.is_known else "UNKNOWN"
			print(f"    [{tag:7s}] {p.prompt}")
