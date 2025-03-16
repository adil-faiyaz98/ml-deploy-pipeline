package test

import (
	"crypto/tls"
	"fmt"
	"testing"
	"time"

	http_helper "github.com/gruntwork-io/terratest/modules/http-helper"
	"github.com/gruntwork-io/terratest/modules/k8s"
	"github.com/gruntwork-io/terratest/modules/terraform"
	test_structure "github.com/gruntwork-io/terratest/modules/test-structure"
)

// TestMLInfrastructure tests the ML deployment pipeline infrastructure
func TestMLInfrastructure(t *testing.T) {
	t.Parallel()

	// Root folder where Terraform code is located
	rootFolder := "../"

	// Deploy to AWS as default for this test (can be parametrized)
	awsRegion := "us-west-2"

	// Undo setup on test completion
	defer test_structure.RunTestStage(t, "teardown", func() {
		terraformOptions := test_structure.LoadTerraformOptions(t, rootFolder)
		terraform.Destroy(t, terraformOptions)
	})

	// Deploy infrastructure
	test_structure.RunTestStage(t, "setup", func() {
		terraformOptions := &terraform.Options{
			TerraformDir: rootFolder,
			Vars: map[string]interface{}{
				"cloud_provider": "aws",
				"environment":    "dev",
				"aws_region":     awsRegion,
			},
		}

		test_structure.SaveTerraformOptions(t, rootFolder, terraformOptions)

		// Deploy the example
		terraform.InitAndApply(t, terraformOptions)
	})

	// Validate infrastructure
	test_structure.RunTestStage(t, "validate", func() {
		terraformOptions := test_structure.LoadTerraformOptions(t, rootFolder)

		// Get the endpoint of the deployed API
		endpoint := terraform.Output(t, terraformOptions, "model_api_endpoint")

		// Wait for API to become available
		url := fmt.Sprintf("https://%s/health", endpoint)
		tlsConfig := tls.Config{}

		// Verify that we get back a 200 OK with the expected text
		http_helper.HttpGetWithRetryWithCustomValidation(
			t,
			url,
			&tlsConfig,
			30,
			10*time.Second,
			func(statusCode int, body string) bool {
				return statusCode == 200 && body == `{"status":"healthy"}`
			},
		)

		// Get kubeconfig
		kubeconfigPath := terraform.Output(t, terraformOptions, "kubeconfig_path")

		// Create a Kubernetes client
		options := k8s.NewKubectlOptions("", kubeconfigPath, "ml-deploy")

		// Check that essential pods are running
		k8s.WaitUntilNumPodsCreated(t, options, k8s.MetaV1ListOptions{
			LabelSelector: "app=model-api",
		}, 2, 5*time.Minute)

		// Verify MLflow pod is running
		mlflowPods := k8s.ListPods(t, options, k8s.MetaV1ListOptions{
			LabelSelector: "app=mlflow",
		})

		if len(mlflowPods) < 1 {
			t.Fatal("Expected at least 1 MLflow pod to be running")
		}
	})
}
