package main

import (
	"encoding/base64"
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"

	"golang.org/x/net/context"
	"golang.org/x/oauth2/google"
	vision "google.golang.org/api/vision/v1"
)

func main() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s <path-to-image>\n", filepath.Base(os.Args[0]))
	}

	flag.Parse()

	args := flag.Args()
	if len(args) == 0 {
		flag.Usage()
		os.Exit(1)
	}

	if err := run(args[0]); err != nil {
		// Comes here if run() returns an error
		fmt.Fprintf(os.Stderr, "%s\n", err.Error())
		os.Exit(1)
	}

}

func run(file string) error {
	ctx := context.Background()

	// Authenticate to generate a vision service
	client, err := google.DefaultClient(ctx, vision.CloudPlatformScope)
	if err != nil {
		return err
	}

	service, err := vision.New(client)
	if err != nil {
		return err
	}
	// We now have a Vision API service with which we can make API calls.

	b, err := ioutil.ReadFile(file)
	if err != nil {
		return err
	}

	// Construct a label request, encoding the image in base64.
	req := &vision.AnnotateImageRequest{
		// Apply image which is encoded by base64
		Image: &vision.Image{
			Content: base64.StdEncoding.EncodeToString(b),
		},
		// Apply features to indicate what type of image detection
		// type: LABEL_DETECTION, FACE_DETECTION, LANDMARK_DETECTION, LOGO_DETECTION, TEXT_DETECTION, DOCUMENT_TEXT_DETECTION, WEB_DETECTION, SAFE_SEARCH_DETECTION
		Features: []*vision.Feature{
			{
				Type:       "LABEL_DETECTION",
				MaxResults: 10,
			},
			{
				Type:       "FACE_DETECTION",
				MaxResults: 10,
			},
			{
				Type:       "LANDMARK_DETECTION",
				MaxResults: 10,
			},
			{
				Type:       "LOGO_DETECTION",
				MaxResults: 10,
			},
			{
				Type:       "TEXT_DETECTION",
				MaxResults: 10,
			},
		},
	}

	batch := &vision.BatchAnnotateImagesRequest{
		Requests: []*vision.AnnotateImageRequest{req},
	}

	res, err := service.Images.Annotate(batch).Do()
	if err != nil {
		return err
	}
	// A POST request has been made

	for r := 0; r < len(res.Responses); r++ {

		// Parse annotations from responses
		if annotations := res.Responses[r].LabelAnnotations; len(annotations) > 0 {
			for i := 0; i < len(annotations); i++ {
				mid := annotations[i].Mid
				label := annotations[i].Description
				score := annotations[i].Score
				fmt.Printf("Found MID: %s\tScore: %f \tLabel: %s\n", mid, score, label)
			}
		} else {
			fmt.Printf("Not found label: %s\n", file)
		}

		if annotations := res.Responses[r].FaceAnnotations; len(annotations) > 0 {
			for i := 0; i < len(annotations); i++ {
				anger := annotations[i].AngerLikelihood
				joy := annotations[i].JoyLikelihood
				sorrow := annotations[i].SorrowLikelihood
				surprise := annotations[i].SurpriseLikelihood
				fmt.Printf("Found Anger: %s\tJoy: %s \tSorrow: %s, Surprise: %s\n", anger, joy, sorrow, surprise)
			}
		} else {
			fmt.Printf("Not found face: %s\n", file)
		}

		if annotations := res.Responses[r].LogoAnnotations; len(annotations) > 0 {
			for i := 0; i < len(annotations); i++ {
				mid2 := annotations[i].Mid
				label2 := annotations[i].Description
				score2 := annotations[i].Score
				fmt.Printf("Found MID: %s\tScore: %f \tLogo: %s\n", mid2, score2, label2)
			}
		} else {
			fmt.Printf("Not found logo: %s\n", file)
		}

		if annotations := res.Responses[r].LandmarkAnnotations; len(annotations) > 0 {
			for i := 0; i < len(annotations); i++ {
				mid3 := annotations[i].Mid
				label3 := annotations[i].Description
				score3 := annotations[i].Score
				fmt.Printf("Found MID: %s\tScore: %f \tLandmark: %s\n", mid3, score3, label3)
			}
		} else {
			fmt.Printf("Not found landmark: %s\n", file)
		}

		if annotations := res.Responses[r].TextAnnotations; len(annotations) > 0 {
			for i := 0; i < len(annotations); i++ {
				mid4 := annotations[i].Mid
				label4 := annotations[i].Description
				score4 := annotations[i].Score
				fmt.Printf("Found MID: %s\tScore: %f \tLabel: %s\n", mid4, score4, label4)
			}
		} else {
			fmt.Printf("Not found text: %s\n", file)
		}
	}

	return nil
}
