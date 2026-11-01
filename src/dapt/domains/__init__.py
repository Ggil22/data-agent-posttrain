"""Training-task domains. None of them shares tables or questions with data-agent-bench."""

from dapt.domains import clinic, energy, inventory, rentals
from dapt.domains.base import Domain

DOMAINS: dict[str, Domain] = {
    d.name: d for d in (rentals.DOMAIN, clinic.DOMAIN, energy.DOMAIN, inventory.DOMAIN)
}
