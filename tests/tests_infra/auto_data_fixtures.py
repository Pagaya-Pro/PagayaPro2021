from tests.tests_infra.common_fixtures import STANDARD_DATA_FOLDER


def get_auto_samples_per_quarter_path(quarter: str, table_type: str):
    """
    Args:
        quarter: In the format "2016Q1"
        table_type: "features" or "targets", in the future can also be more.
    """
    return STANDARD_DATA_FOLDER / "auto_tests_tuples" / f"{quarter}_{table_type}.parquet"


auto_2016q1_tests_tuple_sample100 = (
    get_auto_samples_per_quarter_path("2016Q1", "features"),
    get_auto_samples_per_quarter_path("2016Q1", "targets"),
)

auto_20164_tests_tuple_sample100 = (
    get_auto_samples_per_quarter_path("2016Q4", "features"),
    get_auto_samples_per_quarter_path("2016Q4", "targets"),
)

auto_2017q2_tests_tuple_sample100 = (
    get_auto_samples_per_quarter_path("2017Q2", "features"),
    get_auto_samples_per_quarter_path("2017Q2", "targets"),
)

auto_2018q2_tests_tuple_sample100 = (
    get_auto_samples_per_quarter_path("2018Q2", "features"),
    get_auto_samples_per_quarter_path("2018Q2", "targets"),
)

AUTO_TEST_PLATFORMS_TUPLES_TUPLE = (
    auto_2016q1_tests_tuple_sample100,
    auto_20164_tests_tuple_sample100,
    auto_2017q2_tests_tuple_sample100,
    auto_2018q2_tests_tuple_sample100,
)
