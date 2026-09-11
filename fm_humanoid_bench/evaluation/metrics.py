"""Usage accounting and explicit (never implicit) price estimates."""
FIELDS = ('input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens')

def usage_counts(usage):
    result = {key: usage.get(key,0) for key in FIELDS}
    if any(type(v) is not int or v < 0 for v in result.values()):
        raise ValueError('Token counts must be nonnegative integers')
    if result['cached_input_tokens'] > result['input_tokens']:
        raise ValueError('Cached input cannot exceed total input')
    if result['reasoning_output_tokens'] > result['output_tokens']:
        raise ValueError('Reasoning output cannot exceed total output')
    return result

def estimate_cost(counts, rates):
    """Standard rate estimate; caller supplies model/date/tier provenance."""
    counts = usage_counts(counts)
    if any(float(rates[k]) < 0 for k in ('input','cached_input','output')):
        raise ValueError('Rates cannot be negative')
    return ((counts['input_tokens']-counts['cached_input_tokens'])*rates['input']
            + counts['cached_input_tokens']*rates['cached_input']
            + counts['output_tokens']*rates['output'])/1e6
