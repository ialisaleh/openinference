// import * as grpc from "@grpc/grpc-js";
// import * as uuid from "uuid";
// import * as jsonFormat from "protobufjs";
// import { Session } from "./ArizeSession";
// import {
//   DEFAULT_ARIZE_FLIGHT_HOST,
//   DEFAULT_ARIZE_FLIGHT_PORT,
//   DEFAULT_TRANSPORT_SCHEME,
//   FLIGHT_ACTION_KEY,
// } from "./ArizeConstants";
// import * as request_pb from "./requests_pb"; // Import your proto files here
// import { FlightDescriptor, Ticket, Action } from "apache-arrow-flight"; // Import arrow-flight package
// import * as pandas from "danfojs"; // Use danfo.js for pandas equivalent

// interface ArizeDatasetsClientConfig {
//   developerKey: string;
//   host?: string;
//   port?: number;
//   scheme?: string;
// }

// export class ArizeDatasetsClient {
//   private developerKey: string;
//   private host: string;
//   private port: number;
//   private scheme: string;
//   private session: Session;

//   constructor(config: ArizeDatasetsClientConfig) {
//     this.developerKey = config.developerKey;
//     this.host = config.host || DEFAULT_ARIZE_FLIGHT_HOST;
//     this.port = config.port || DEFAULT_ARIZE_FLIGHT_PORT;
//     this.scheme = config.scheme || DEFAULT_TRANSPORT_SCHEME;

//     this.session = new Session({
//       developerKey: this.developerKey,
//       host: this.host,
//       port: this.port,
//       scheme: this.scheme,
//     });
//   }

//   public async getDataset(
//     spaceId: string,
//     datasetId?: string,
//     datasetName?: string,
//     datasetVersion?: string,
//     convertJsonStrToDict = true,
//   ): Promise<pandas.DataFrame | null> {
//     if (!(datasetId || datasetName)) {
//       throw new Error(
//         `One and only one of datasetId=${datasetId} or datasetName=${datasetName} is required.`,
//       );
//     }

//     const request = new request_pb.DoGetRequest({
//       getDataset: new request_pb.GetDatasetRequest({
//         space_id: spaceId,
//         dataset_version: datasetVersion || "",
//         dataset_id: datasetId,
//         dataset_name: datasetName,
//       }),
//     });

//     const ticket = this._ticketForRequest(request);
//     try {
//       const flightClient = this.session.connect();
//       const reader = await flightClient.doGet(ticket, this.session.metadata);
//       const df = pandas.readParquet(reader); // Assuming the data comes in parquet format
//       if (convertJsonStrToDict) {
//         return this._convertJsonStrToDict(df);
//       }
//       return df;
//     } catch (error) {
//       throw new Error(
//         `Failed to get dataset name=${datasetName}, id=${datasetId} for spaceId=${spaceId}. Error: ${error}`,
//       );
//     }
//   }

//   private _descriptorForRequest(request: grpc.Message): FlightDescriptor {
//     const data = jsonFormat.MessageToJson(request).toString();
//     return FlightDescriptor.forCommand(data);
//   }

//   private _ticketForRequest(request: grpc.Message): Ticket {
//     const data = jsonFormat.MessageToJson(request).toString();
//     return new Ticket({ ticket: Buffer.from(data).toString() });
//   }

//   private _actionForRequest(
//     actionKey: FLIGHT_ACTION_KEY,
//     request: grpc.Message,
//   ): Action {
//     const reqBytes = jsonFormat.MessageToJson(request).toString();
//     return new Action(actionKey.value, Buffer.from(reqBytes));
//   }

//   private _convertJsonStrToDict(df: pandas.DataFrame): pandas.DataFrame {
//     df.columns.forEach((col: string) => {
//       if (this._shouldConvert(col)) {
//         df[col] = df[col].apply((x: string) => {
//           try {
//             return JSON.parse(x);
//           } catch (error) {
//             console.error(`Failed to convert column ${col} to JSON`);
//             return x;
//           }
//         });
//         console.log(`Converted column ${col} to dict for data export`);
//       }
//     });
//     return df;
//   }

//   private _shouldConvert(colName: string): boolean {
//     const isEvalMetadata =
//       colName.startsWith("eval.") && colName.endsWith(".metadata");
//     const isJsonStr = OPEN_INFERENCE_JSON_STR_TYPES.includes(colName);
//     return isEvalMetadata || isJsonStr;
//   }

//   private _setDefaultColumnsForDataset(df: pandas.DataFrame): pandas.DataFrame {
//     const currentTime = Date.now();

//     if (df.columns.includes("created_at")) {
//       df["created_at"] = df["created_at"].fillna(currentTime);
//     } else {
//       df["created_at"] = currentTime;
//     }

//     if (df.columns.includes("updated_at")) {
//       df["updated_at"] = df["updated_at"].fillna(currentTime);
//     } else {
//       df["updated_at"] = currentTime;
//     }

//     if (df.columns.includes("id")) {
//       df["id"] = df["id"].apply((x: any) => (x ? x : uuid.v4()));
//     } else {
//       df["id"] = df.apply(() => uuid.v4());
//     }

//     return df;
//   }

//   private _convertDefaultColumnsToJsonStr(
//     df: pandas.DataFrame,
//   ): pandas.DataFrame {
//     df.columns.forEach((col: string) => {
//       if (this._shouldConvert(col)) {
//         df[col] = df[col].apply((x: any) => JSON.stringify(x));
//         console.log(`Converted ${col} to JSON string for data import`);
//       }
//     });
//     return df;
//   }
// }
