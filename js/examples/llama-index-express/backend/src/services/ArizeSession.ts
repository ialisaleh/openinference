import path from "path";

import * as uuid from "uuid";
import * as grpc from "@grpc/grpc-js";
import * as protoLoader from "@grpc/proto-loader";

import { DEFAULT_PACKAGE_NAME, InvalidSessionError } from "./ArizeConstants";

interface SessionConfig {
  developerKey: string;
  host: string;
  port: number;
  scheme: string;
}

export class Session {
  private developerKey: string;
  private host: string;
  private port: number;
  private scheme: string;
  private sessionName: string;
  private headers: { [key: string]: string };

  private flightProto: any;
  private client: any;
  private metadata: grpc.Metadata;

  constructor(config: SessionConfig) {
    this.developerKey = config.developerKey;
    this.host = config.host;
    this.port = config.port;
    this.scheme = config.scheme;
    this.sessionName = `typescript-sdk-${DEFAULT_PACKAGE_NAME}-${uuid.v4()}`;
    console.debug(`Creating named session as '${this.sessionName}'.`);

    if (!this.developerKey) {
      console.error(InvalidSessionError.errorMessage());
      throw new InvalidSessionError("Developer Key Not Found.");
    }

    console.debug(
      `Created session with Arize Developer Key '${this.developerKey}' at '${this.host}':'${this.port}'`,
    );

    this.setHeaders();
    this.loadProto();
  }

  private setHeaders(): void {
    this.headers = {
      "origin": "arize-typescript-datasets-client",
      "auth-token-bin": this.developerKey,
      "sdk-language": "typescript",
      "language-version": process.version,
      "sdk-version": "1.0.0",
    };
  }

  private loadProto(): void {
    const PROTO_PATH = path.join(__dirname, "requests.proto");

    const packageDefinition = protoLoader.loadSync(PROTO_PATH, {
      keepCase: true,
      longs: String,
      enums: String,
      defaults: true,
      oneofs: true,
    });

    const protoDescriptor = grpc.loadPackageDefinition(packageDefinition);
    this.flightProto = protoDescriptor.public; // Replace with the actual service package name.
  }

  public connect(): grpc.Client {
    try {
      const url = `${this.scheme}://${this.host}:${this.port}`;
      const client = new (this.flightProto
        .DatasetService as grpc.ServiceClientConstructor)(
        url,
        grpc.credentials.createInsecure(),
      );

      // Add headers/metadata for calls.
      const metadata = new grpc.Metadata();
      for (const [key, value] of Object.entries(this.headers)) {
        metadata.add(key, value);
      }

      this.client = client;
      this.metadata = metadata;
      return client;
    } catch (error) {
      console.error(
        "There was an error trying to connect to the Arize Flight Endpoint",
      );
      throw error;
    }
  }

  // Example function for dataset retrieval
  public async getDataset(
    spaceId: string,
    datasetId?: string,
    datasetName?: string,
    datasetVersion?: string,
  ): Promise<any> {
    // Validation to ensure one of datasetId or datasetName is provided
    if (!(datasetId || datasetName)) {
      throw new Error("You must provide either datasetId or datasetName.");
    }

    // Create the request using protobuf
    const request = {
      get_dataset: {
        space_id: spaceId,
        dataset_version: datasetVersion || "",
        dataset_id: datasetId,
        dataset_name: datasetName,
      },
    };

    const ticket = this._createTicketForRequest(request);

    // Call the Flight service
    return new Promise((resolve, reject) => {
      this.client.doGet(
        ticket,
        this.metadata,
        (error: grpc.ServiceError, response: any) => {
          if (error) {
            reject(error);
          } else {
            // Process the response and convert to DataFrame or desired format
            const data = response.toPandas(); // Adjust accordingly
            resolve(data);
          }
        },
      );
    });
  }

  // Helper method to create a Flight Ticket
  private _createTicketForRequest(request: any): any {
    const requestJson = JSON.stringify(request);
    return { ticket: requestJson }; // Adjust based on your proto structure
  }
}
