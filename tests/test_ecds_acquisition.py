import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from data_audit.ecds import run
from data_audit.ecds.decode import compare, decode
from data_audit.audit import HERE

try:
    import eccodes as ec
except ImportError:
    ec = None


def encoded_fixture(path, ref):
    """Exercise the native decoder with real GRIB2 messages and controlled values."""
    fields = {}
    for r in ref["rows"]:
        for param, var in [(165, "u10"), (166, "v10")]:
            fields.setdefault((r["lead_hours"], r["member"], param), {})[r["lat"], r["lon"]] = r[var]
    h = ec.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    try:
        for key, val in dict(centre="ecmf", productDefinitionTemplateNumber=1,
                             typeOfProcessedData=4, typeOfEnsembleForecast=3,
                             numberOfForecastsInEnsemble=50, dataDate=20190824, dataTime=0,
                             Ni=2, Nj=2, latitudeOfFirstGridPointInDegrees=57.5,
                             longitudeOfFirstGridPointInDegrees=356,
                             latitudeOfLastGridPointInDegrees=57,
                             longitudeOfLastGridPointInDegrees=356.5,
                             iDirectionIncrementInDegrees=0.5, jDirectionIncrementInDegrees=0.5,
                             bitsPerValue=16).items():
            ec.codes_set(h, key, val)
        with path.open("wb") as stream:
            for (lead, member, param), values in reversed(fields.items()):
                for key, val in dict(forecastTime=lead, perturbationNumber=member, paramId=param).items():
                    ec.codes_set(h, key, val)
                ec.codes_set_values(h, [values[lat, lon] for lat in (57.5, 57) for lon in (-4, -3.5)])
                ec.codes_write(h, stream)
    finally:
        ec.codes_release(h)


class AcquisitionTests(unittest.TestCase):
    def test_auth_failure_stops_without_anonymous_submission(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(
            run, "client", side_effect=run.Stop("credentials_not_configured")
        ):
            status = run.run(Path(folder))
        self.assertEqual(len(status["days"]), 30)
        self.assertEqual(status["days"][-1]["day"], "2019-09-22")
        self.assertEqual(status["downloaded_body_bytes_this_run"], 0)
        self.assertFalse(any(d["request_submitted_this_run"] for d in status["days"]))
        self.assertEqual(status["days"][1]["status"], "not_attempted")

    def test_remote_errors(self):
        for code, message, expected in [(401, "", "authentication_rejected"),
                                       (403, "licences not accepted", "licence_rejected"),
                                       (403, "", "permission_denied"),
                                       (422, "", "request_parameters_rejected")]:
            error = RuntimeError(message)
            error.response = Mock(status_code=code)
            self.assertEqual(run.classify(error), expected)

    def test_oversize(self):
        with self.assertRaisesRegex(run.Stop, "outside_limit"):
            run.download(None, None, run.LIMIT, {}, run.LIMIT + 1)

    @unittest.skipUnless(ec, "Requires optional acquisition environment with ecCodes")
    def test_native_content_and_replay(self):
        reference = json.loads((HERE / "ens.json").read_bytes())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root / "synthetic.grib"
            encoded_fixture(raw, reference)
            candidate, bounds = decode(raw, run.DAYS[0])
            self.assertTrue(all(m["outside"] == 0 for m in compare(reference, candidate, bounds).values()))
            candidate["rows"][0]["v10"] += 1
            self.assertEqual(compare(reference, candidate, bounds)["v10"]["outside"], 1)
            with self.assertRaises(ValueError):
                decode(raw, run.DAYS[1])
            service = Mock()
            service.retrieve.side_effect = [Mock(location=raw.as_uri(), content_length=raw.stat().st_size),
                                            run.Stop("test_stop")]
            with patch.object(run, "client", return_value=service):
                ledger = run.run(root)
            self.assertEqual(ledger["accepted_windows"], 1)
            self.assertEqual(ledger["downloaded_body_bytes_this_run"], raw.stat().st_size)
            with patch.object(run, "client", side_effect=run.Stop("credentials_not_configured")):
                replay = run.run(root)
            self.assertEqual(replay["accepted_windows"], 1)
            self.assertEqual(replay["downloaded_body_bytes_this_run"], 0)
            self.assertFalse(replay["dataset_train_ready"])
