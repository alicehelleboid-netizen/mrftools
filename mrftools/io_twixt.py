# import twixtools


def extract_voi(filename, meas_index=None):
    headers = parse_twixt_header(filename)

    # filter headers for geometry
    headers = [header for header in headers if has_geometry(header)]

    if not headers:
        raise ValueError("No suitable acquisitions were found")

    geometries = []
    names = []
    for header in headers:
        geometry = get_geometry(header)
        if geometry in geometries:
            continue
        names.append(header.get("tProtocolName"))
        geometries.append(geometry)

    if meas_index is not None:
        return geometries[meas_index]
    elif len(geometries) == 1:
        return geometries[0]
    else:
        return geometries


def has_geometry(header):
    return "sAdjData.sAdjVolume.dReadoutFOV" in header


def get_geometry(header):
    """extract geometry from ASCCONV header"""
    # VOI center (sagital, coronal, transversal)
    position = [
        header.get("sAdjData.sAdjVolume.sPosition.dSag", 0),
        header.get("sAdjData.sAdjVolume.sPosition.dCor", 0),
        header.get("sAdjData.sAdjVolume.sPosition.dTra", 0),
    ]
    size = [
        header["sAdjData.sAdjVolume.dReadoutFOV"],
        header["sAdjData.sAdjVolume.dPhaseFOV"],
        header["sAdjData.sAdjVolume.dThickness"],
    ]
    normal = [
        header.get("sAdjData.sAdjVolume.sNormal.dSag", 0),
        header.get("sAdjData.sAdjVolume.sNormal.dCor", 0),
        header.get("sAdjData.sAdjVolume.sNormal.dTra", 0),
    ]
    rotation = header.get("sAdjData.sAdjVolume.dInPlaneRot", 0)
    name = header.get("tProtocolName", "Unknown")

    # VOI info
    return {
        "position": position,
        "size": size,
        "normal": normal,
        "rotation": rotation,
        "name": name,
    }


def parse_twixt_header(filename):
    with open(filename, "rb") as fp:
        data = fp.read()
    data = data.decode("latin1")
    begin = "### ASCCONV BEGIN"
    end = "### ASCCONV END ###"
    split = data.replace(end, begin).split(begin)[1::2]
    headers = []
    for item in split:
        header = {}
        for line in item.split("\n"):
            line = line.strip()
            if not line:
                continue
            key, value = line.split("=", 1)
            header[key.strip()] = parse_value(value)
        headers.append(header)
    return headers


def parse_value(value):
    """parse string value into number if possible"""
    # remove comments
    value = value.split("#", 1)[0].strip()

    if value[:2] == "0x":
        return int(value, 0)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.replace("'", "").replace('"', "")


# def extract_voi(twixt_file, meas_index=-1):

#     # read twixt file
#     opts = dict(
#         include_scans=[meas_index],
#         parse_data=False,
#         parse_geometry=False,
#     )
#     scan = twixtools.read_twix(str(twixt_file), **opts)[0]

#     # extract relevant information
#     hdr = scan['hdr']
#     info = hdr['MeasYaps']['sAdjData']['sAdjVolume']

#     # VOI center (sagital, coronal, transversal)
#     position = [info['sPosition'].get('dSag', 0), info['sPosition'].get('dCor', 0), info['sPosition'].get('dTra', 0)]
#     size = [info['dReadoutFOV'], info['dPhaseFOV'], info['dThickness']]
#     normal = [info['sNormal'].get('dSag', 0), info['sNormal'].get('dCor', 0), info['sNormal'].get('dTra', 0)]
#     rotation = info['dInPlaneRot']

#     # VOI info
#     return {
#         'position': position,
#         'size': size,
#         'normal': normal,
#         'rotation': rotation,
#     }
