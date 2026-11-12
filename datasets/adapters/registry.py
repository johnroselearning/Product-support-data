from . import it_support, helpdesk
ADAPTERS = {"it_support": it_support.adapt, "helpdesk": helpdesk.adapt}
def get_adapter(source_id):
    try:
        return ADAPTERS[source_id]
    except KeyError as exc:
        raise ValueError(f"No adapter implemented for enabled source: {source_id}") from exc
