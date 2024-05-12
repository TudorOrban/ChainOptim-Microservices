def to_camel(string: str) -> str:
    if not isinstance(string, str):
        return string 
    components = string.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])
