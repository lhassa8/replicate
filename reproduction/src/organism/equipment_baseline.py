"""Development fixed policy through actual reader tools, with no oracle access.

This baseline is a candidate awaiting review, not the frozen Q5 baseline.
It inspects published descriptors; it has no family IDs, fixture IDs, expected
answers or lookup table of private reader selections.
"""


def solve(session, device_id):
    cap = session.inspect_equipment(device_id)["descriptor"]
    target_unit = cap["task"]["canonical_unit"]
    allowed = {"C": {"C", "F", "K"}, "mm/s": {"mm/s", "in/s"},
               "percent": {"ratio", "percent"}, "count": {"count"}}[target_unit]
    candidates = []
    for source in cap["sources"]:
        values = [field for field in source["fields"] if field["kind"] == "number" and field["unit"] in allowed]
        times = [field for field in source["fields"] if field["kind"] == "timestamp"]
        validity = [field for field in source["fields"] if field["kind"] == "validity"]
        if len(times) > 1:
            times = [field for field in times if field["meaning"].lower() in (
                "measurement sampling time", "sensor sampling time", "observation window end time")]
        if len(values) == 1 and len(times) == 1 and len(validity) <= 1:
            candidates.append((source["source_id"], values[0]["selector"], times[0]["selector"],
                               validity[0]["selector"] if validity else None))
    if len(candidates) != 1:
        raise ValueError("Fixed policy cannot select an unambiguous supported reader")
    source, value_selector, timestamp, validity = candidates[0]
    session.configure_reader(device_id, source, [value_selector], timestamp, validity)
    reading = session.read_equipment(device_id)
    value = reading["values"][0]
    unknown = (reading["validity"] is not True or not reading["timestamp_usable"] or not value["available"]
               or reading["age_seconds"] > cap["task"]["maximum_age_seconds"])
    result, status = None, "unknown"
    if not unknown:
        result = value["value"] * value["scale"]
        source_unit = value["unit"]
        if source_unit == "F":
            result = (result - 32) * 5 / 9
        elif source_unit == "K":
            result -= 273.15
        elif source_unit == "in/s":
            result *= 25.4
        elif source_unit == "ratio":
            result *= 100
        low, high = cap["task"]["normal_range"]
        status = "low" if result < low else "high" if result > high else "normal"
    session.submit_diagnosis(device_id, status, result, target_unit, reading["evidence_id"])
    return session.submission
