"""Canonical Identity Resolution and Alias Mapping Engine."""

import re
import unicodedata
from sqlalchemy.orm import Session
from src.db.models import Participant, ParticipantAlias

COMMON_ALIASES = {
    "football": {
        "man utd": "Manchester United",
        "manchester united": "Manchester United",
        "manchester united fc": "Manchester United",
        "man united": "Manchester United",
        "man city": "Manchester City",
        "manchester city": "Manchester City",
        "liverpool": "Liverpool",
        "liverpool fc": "Liverpool",
        "arsenal": "Arsenal",
        "arsenal fc": "Arsenal",
        "chelsea": "Chelsea",
        "chelsea fc": "Chelsea",
        "spurs": "Tottenham Hotspur",
        "tottenham": "Tottenham Hotspur",
        "tottenham hotspur": "Tottenham Hotspur",
        "newcastle": "Newcastle United",
        "newcastle united": "Newcastle United",
        "real madrid": "Real Madrid",
        "barcelona": "Barcelona",
        "fc barcelona": "Barcelona",
        "atletico madrid": "Atletico Madrid",
        "bayern munich": "Bayern Munich",
        "bayern munchen": "Bayern Munich",
        "borussia dortmund": "Borussia Dortmund",
        "dortmund": "Borussia Dortmund",
        "juventus": "Juventus",
        "inter": "Inter Milan",
        "internazionale": "Inter Milan",
        "milan": "AC Milan",
        "ac milan": "AC Milan",
        "psg": "Paris Saint-Germain",
        "paris sg": "Paris Saint-Germain"
    },
    "basketball": {
        "atl": "Atlanta Hawks",
        "atlantahawks": "Atlanta Hawks",
        "bos": "Boston Celtics",
        "celtics": "Boston Celtics",
        "bkn": "Brooklyn Nets",
        "cha": "Charlotte Hornets",
        "chi": "Chicago Bulls",
        "bulls": "Chicago Bulls",
        "cle": "Cleveland Cavaliers",
        "dal": "Dallas Mavericks",
        "den": "Denver Nuggets",
        "det": "Detroit Pistons",
        "gsw": "Golden State Warriors",
        "warriors": "Golden State Warriors",
        "hou": "Houston Rockets",
        "ind": "Indiana Pacers",
        "lac": "LA Clippers",
        "los angeles clippers": "LA Clippers",
        "lal": "Los Angeles Lakers",
        "lakers": "Los Angeles Lakers",
        "mem": "Memphis Grizzlies",
        "mia": "Miami Heat",
        "mil": "Milwaukee Bucks",
        "min": "Minnesota Timberwolves",
        "nop": "New Orleans Pelicans",
        "nyk": "New York Knicks",
        "knicks": "New York Knicks",
        "okc": "Oklahoma City Thunder",
        "orl": "Orlando Magic",
        "phi": "Philadelphia 76ers",
        "sixers": "Philadelphia 76ers",
        "phx": "Phoenix Suns",
        "por": "Portland Trail Blazers",
        "sac": "Sacramento Kings",
        "sas": "San Antonio Spurs",
        "tor": "Toronto Raptors",
        "uta": "Utah Jazz",
        "was": "Washington Wizards"
    },
    "tennis": {
        "novak djokovic": "Novak Djokovic",
        "djokovic n.": "Novak Djokovic",
        "n. djokovic": "Novak Djokovic",
        "carlos alcaraz": "Carlos Alcaraz",
        "alcaraz c.": "Carlos Alcaraz",
        "jannik sinner": "Jannik Sinner",
        "sinner j.": "Jannik Sinner",
        "daniil medvedev": "Daniil Medvedev",
        "medvedev d.": "Daniil Medvedev",
        "alexander zverev": "Alexander Zverev",
        "zverev a.": "Alexander Zverev",
        "rafael nadal": "Rafael Nadal",
        "nadal r.": "Rafael Nadal",
        "roger federer": "Roger Federer",
        "federer r.": "Roger Federer",
        "iga swiatek": "Iga Swiatek",
        "swiatek i.": "Iga Swiatek",
        "aryna sabalenka": "Aryna Sabalenka",
        "sabalenka a.": "Aryna Sabalenka",
        "coco gauff": "Coco Gauff",
        "gauff c.": "Coco Gauff"
    }
}


def clean_name_string(raw_name: str) -> str:
    """Normalize whitespace, unicode accents, and lowercase for key lookup."""
    if not raw_name:
        return ""
    text = str(raw_name).strip()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def slugify_entity(canonical_name: str, sport: str) -> str:
    clean = clean_name_string(canonical_name).lower()
    clean = re.sub(r"[^a-z0-9]+", "_", clean).strip("_")
    return f"{sport}_{clean}"


def resolve_canonical_participant(
    db: Session,
    raw_name: str,
    sport_id: str,
    entity_type: str = "TEAM",
    source: str = "SYSTEM"
) -> Participant:
    """Resolve a raw participant string into a canonical Participant entity."""
    clean = clean_name_string(raw_name)
    lookup_key = clean.lower()

    # 1. Check known built-in dictionary
    canonical_display = COMMON_ALIASES.get(sport_id, {}).get(lookup_key, clean)

    # 2. Check database alias table
    existing_alias = (
        db.query(ParticipantAlias)
        .filter(ParticipantAlias.alias_name == lookup_key)
        .first()
    )
    if existing_alias:
        return existing_alias.participant

    # 3. Check if canonical participant already exists
    participant_id = slugify_entity(canonical_display, sport_id)
    participant = db.query(Participant).filter(Participant.participant_id == participant_id).first()

    if not participant:
        participant = Participant(
            participant_id=participant_id,
            sport_id=sport_id,
            entity_type=entity_type,
            canonical_name=canonical_display
        )
        db.add(participant)
        db.flush()

    # 4. Save new alias
    new_alias = ParticipantAlias(
        participant_id=participant.participant_id,
        alias_name=lookup_key,
        source=source
    )
    db.add(new_alias)
    db.commit()

    return participant
