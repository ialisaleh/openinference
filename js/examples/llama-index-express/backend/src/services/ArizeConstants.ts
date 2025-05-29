import { Enum } from "typescript-string-enums";

// Arize Dataset Types
const UNKNOWN = 0;
const GENERATIVE = 1;
const INFERENCES = 2;

// Default API endpoint when not provided through env variable nor profile.
const DEFAULT_ARIZE_FLIGHT_HOST = "flight.arize.com";
const DEFAULT_ARIZE_FLIGHT_PORT = 443;

// Name of the current package.
const DEFAULT_PACKAGE_NAME = "arize_typescript_datasets_client";

// Default config keys for the Arize config file. Created via the CLI.
const DEFAULT_ARIZE_API_KEY_CONFIG_KEY = "api_key";

// Default headers to trace and help identify requests. For debugging.
const DEFAULT_ARIZE_SESSION_ID = "x-arize-session-id"; // Generally the session name.
const DEFAULT_ARIZE_TRACE_ID = "x-arize-trace-id";
const DEFAULT_PACKAGE_VERSION = "x-package-version";

// Default initial wait time for retries in seconds.
const DEFAULT_RETRY_INITIAL_WAIT_TIME = 0.25;

// Default maximum wait time for retries in seconds.
const DEFAULT_RETRY_MAX_WAIT_TIME = 10.0;

// Default to use grpc + tls scheme.
const DEFAULT_TRANSPORT_SCHEME = "grpc+tls";

const FLIGHT_ACTION_KEY = Enum({
  GET_DATASET_VERSION: "get_dataset_version",
  LIST_DATASETS: "list_datasets",
  DELETE_DATASET: "delete_dataset",
});
type FLIGHT_ACTION_KEY = Enum<typeof FLIGHT_ACTION_KEY>;

class DatasetError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DatasetError";
  }
}

class InvalidSessionError extends DatasetError {
  static errorMessage(): string {
    return (
      "Credentials not provided or invalid. Please pass in the correct api_key when " +
      "initiating a new ArizeExportClient. Alternatively, you can set up credentials " +
      "in a profile or as an environment variable."
    );
  }
}

export {
  UNKNOWN,
  GENERATIVE,
  INFERENCES,
  DEFAULT_ARIZE_FLIGHT_HOST,
  DEFAULT_ARIZE_FLIGHT_PORT,
  DEFAULT_PACKAGE_NAME,
  DEFAULT_ARIZE_API_KEY_CONFIG_KEY,
  DEFAULT_ARIZE_SESSION_ID,
  DEFAULT_ARIZE_TRACE_ID,
  DEFAULT_PACKAGE_VERSION,
  DEFAULT_RETRY_INITIAL_WAIT_TIME,
  DEFAULT_RETRY_MAX_WAIT_TIME,
  DEFAULT_TRANSPORT_SCHEME,
  FLIGHT_ACTION_KEY,
  InvalidSessionError,
};
