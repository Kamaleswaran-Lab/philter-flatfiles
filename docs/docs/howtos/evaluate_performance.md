# How to Evaluate Philter's Performance

A common question we receive is how well does Philter perform? Our answer to this question is probably less than satisfactory because it simply depends. What does it depend on? Philter's performance is heavily dependent upon your individual data. Sharing to compare metrics of Philter's performance between different customer datasets is like comparing apples and oranges.

If your data is not exactly like another customer's data then the metrics will not be applicable to your data. In terms of the classic information retrieval metrics precision and recall, comparing these values between customers can give false impressions about Philter's performance, both good and bad.

> This guide walks you through how to evaluate Philter's performance. If you are just getting started with Philter please see the Quick Starts instead. Then you can come back here to learn how to evaluate Philter's performance.


## Guide to Evaluating Performance

We have created this guide to help guide you in evaluating Philter's performance on your data. The guide involves determining the types of sensitive information you want to redact, configuring those filters, optimizing the configuration, and then capturing the performance metrics.

> We will gladly perform these steps for you and provide you a detailed Philter performance report generated from your data. Please contact us to start the process.


#### What You Need

To evaluate Philter's performance you need:

* A running instance of Philter. If you do not yet have a running instance of Philter you can [launch one](https://www.philterd.ai/philter).
* A list of the types of sensitive information you want to redact.
* A data set representative of the text you will be redacting using Philter. It's important the data set be representative so the evaluation results will transfer to the actual data redaction.
* The same data set but with annotated sensitive information. These annotations will be used to calculate the precision and recall metrics.

#### Configuring Philter

Before we can begin our evaluation we need to create a policy. A [policy](policies_README.md) is a file that defines the types of sensitive information that will be redacted and how it will be redacted. The policies are stored on the Philter instance under `/opt/philter/policies`. You can edit the policies directly there using a text editor or you can use Philter's [API](policies-api.md) to upload a policy. In this case we recommend just using a text editor on the Philter instance to create a policy.

When using a text editor to create and edit a policy, be sure to save the policy often. Frequent saving can make editing a policy easier.

We also recommend considering to place your policy directory under source control to have a history and change log of your policies.

#### Creating a Policy

Make a copy of the default policy and we will modify the copy for our needs.

`cp /opt/philter/policies/default.json /opt/philter/policies/evaluation.json`

Now open `/opt/philter/policies/evaluation.json` in a text editor. (The content of `evaluation.json` will be similar to what's shown below but may have minor differences between different versions of Philter.)

```
{
   "name": "default",
   "identifiers": {
      "emailAddress": {
         "emailAddressFilterStrategies": [
            {
               "strategy": "REDACT",
               "redactionFormat": "{{{REDACTED-%t}}}"
            }
         ]
      },
      "phoneNumber": {
         "phoneNumberFilterStrategies": [
            {
               "strategy": "REDACT",
               "redactionFormat": "{{{REDACTED-%t}}}"
            }
         ]
      }
   }
}
```

The first thing we need to do is to set the name of the policy. Replace `default` with `evaluation` and save the file.

#### Identifying the Filters You Need

The rest of the file contains the filters that are enabled in the default policy. We need to make sure that each type of sensitive information that you want to redact is represented by a filter in this file. Look through the rest of the policy and determine which filters are listed that you do not need and also which filters you do need that are not listed.

#### Disabling Filters We Do Not Need

If a filter is listed in the policy and you do not need the filter you have two options. You can either delete those lines from the policy and save the file, or you can set the filter's `enabled` property to false. Using the `enabled` property allows you to keep the filter configuration in the policy in case it is needed later but both options have the same effect.

#### Enabling Filters Not in the Default Policy

Let's say you want to redact bitcoin addresses. The bitcoin address filter is not in the default policy. To add the bitcoin address filter we will refer to Philter's documentation on the bitcoin address filter, get the configuration, and copy it into the policy.

From the [bitcoin address filter documentation](bitcoin-addresses.md) we see the configuration for the bitcoin address filter is:

```
      "bitcoinAddress": {
         "bitcoinAddressFilterStrategies": [
            {
               "strategy": "REDACT",
               "redactionFormat": "{{{REDACTED-%t}}}"
            }
         ]
      }
```

We can copy this configuration and paste it into our policy:

```
{
   "name": "evaluation",
   "identifiers": {
      "bitcoinAddress": {
         "bitcoinAddressFilterStrategies": [
            {
               "strategy": "REDACT",
               "redactionFormat": "{{{REDACTED-%t}}}"
            }
         ]
      },
      "emailAddress": {
         "emailAddressFilterStrategies": [
            {
               "strategy": "REDACT",
               "redactionFormat": "{{{REDACTED-%t}}}"
            }
         ]
      },
      "phoneNumber": {
         "phoneNumberFilterStrategies": [
            {
               "strategy": "REDACT",
               "redactionFormat": "{{{REDACTED-%t}}}"
            }
         ]
      }
   }
}
```

The order of the filters in the policy does not matter and has no impact on performance. We typically place the filters in the policy alphabetically just to improve readability.

Repeat these steps until you have added a filter for each of the types of sensitive information you want to redact. Typically, the default redaction `strategy` and `redactionFormat` values for each filter should be fine for evaluation.

When finished modifying the policy, save the file and close the text editor. Now restart Philter for the policy changes to be loaded:

```
sudo systemctl restart philter
```

#### Submitting Text for Redaction

With our policy in place we can now send text to Philter for redaction using that policy:

```
curl -k -X POST "https://localhost:8080/api/filter?p=evaluation" -d @file.txt -H "Content-Type: text/plain"
```

In the command above, we are sending the file `file.txt` to Philter. The `?p=evaluation` tells Philter to apply the `evaluation` policy that we have been editing. Philter's response to this command will be the redacted contents of `file.txt` as defined in the policy.

#### Comparing Documents

With the original document `file.txt` and the redacted contents returned by Philter, we can now compare those files to begin evaluating Philter's performance. You can `diff` the text to find the redacted information or use some other method.

A visual comparison provides a quick overview of how Philter is performing on your text but does not give us precision and recall metrics. To calculate these metrics we must compare the redacted document with an annotated file instead of the original file. The annotated file should have the same contents of the original file but with the sensitive information denoted or somehow marked.

There are many industry-standard ways to annotate text and many tools to assist with text annotation. We recommend using a tool to help you annotate and compare instead of performing only a visual comparison which does not provide metric values.

Let's resubmit the file to Philter but instead this time use the explain API `endpoint`:

```
curl -k -X POST "https://localhost:8080/api/explain?p=evaluation" -d @file.txt -H "Content-Type: text/plain"
```

The `explain` API [endpoint](filtering-api.md#explain) produces a detailed description of the redaction. The response will include a list of spans that contain the start and stop positions of redacted text and the type of sensitive information that was redacted. Using this information we can compare the redacted information to our annotated file to calculate precision and recall metrics.

#### Calculating Precision and Recall

Now we can calculate the precision and recall metrics.

* Precision is the number of true positives divided by the number true positives plus false positives.
* Recall is the number of true positives divided by the number of false negatives plus true positives.

![Calculating the precision and recall](../img/precision.png)

* The F-1 score is the harmonic mean of precision and recall.

![Calculating the F-1 score](../img/f1.png)
